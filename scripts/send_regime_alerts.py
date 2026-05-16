#!/usr/bin/env python3
"""Send regime change alerts after a dashboard data build (Phase 6).

Typical usage (run automatically after build_dashboard_data.py):

    python scripts/send_regime_alerts.py

    # Dry run — print what would be sent, send nothing
    python scripts/send_regime_alerts.py --dry-run

    # Manage subscriptions
    python scripts/send_regime_alerts.py --add-email trader@example.com
    python scripts/send_regime_alerts.py --add-webhook https://hooks.slack.com/...
    python scripts/send_regime_alerts.py --remove-email trader@example.com
    python scripts/send_regime_alerts.py --list

SMTP env vars (required for email delivery):
    ALERT_SMTP_HOST   smtp.gmail.com
    ALERT_SMTP_PORT   587
    ALERT_SMTP_USER   you@gmail.com
    ALERT_SMTP_PASS   <gmail-app-password>
    ALERT_FROM_EMAIL  Regime Monitor <you@gmail.com>   (optional)

The script reads:
    dashboard/public/data/summary_prev.json   — previous build snapshot
    dashboard/public/data/summary.json        — current build snapshot

If summary_prev.json is missing (first run after a cold start), no alerts
are sent (no baseline to compare against). The script exits 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts.change_detector import detect_regime_changes  # noqa: E402
from src.alerts.dispatcher import (  # noqa: E402
    add_subscriber,
    count_active_subscribers,
    dispatch_alerts,
    load_subscribers,
    remove_subscriber,
)

DATA_DIR = ROOT / "dashboard" / "public" / "data"
CURR_PATH = DATA_DIR / "summary.json"
PREV_PATH = DATA_DIR / "summary_prev.json"
STORE_PATH = ROOT / "data" / "subscribers.json"


def _load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _build_smtp_config() -> dict | None:
    host = os.environ.get("ALERT_SMTP_HOST")
    user = os.environ.get("ALERT_SMTP_USER")
    password = os.environ.get("ALERT_SMTP_PASS")
    if not (host and user and password):
        return None
    return {
        "host": host,
        "port": int(os.environ.get("ALERT_SMTP_PORT", 587)),
        "user": user,
        "password": password,
        "from_email": os.environ.get("ALERT_FROM_EMAIL", user),
    }


def _build_resend_config() -> dict | None:
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        return None
    return {
        "api_key": api_key,
        "from_email": os.environ.get(
            "ALERT_FROM_EMAIL", "Regime Monitor <alerts@resend.dev>"
        ),
    }


def cmd_send(args: argparse.Namespace) -> int:
    if not CURR_PATH.exists():
        print(f"[alerts] {CURR_PATH} not found — run build_dashboard_data.py first", file=sys.stderr)
        return 1

    if not PREV_PATH.exists():
        print("[alerts] no previous snapshot (summary_prev.json) — skipping on first run")
        return 0

    prev = _load_json(PREV_PATH)
    curr = _load_json(CURR_PATH)

    report = detect_regime_changes(prev, curr)

    if not report["has_changes"]:
        print("[alerts] no regime changes detected — nothing to send")
        return 0

    n_asset = len(report["asset_changes"])
    n_cons = 1 if report["consensus_change"] else 0
    print(f"[alerts] detected {n_asset} asset change(s), {n_cons} consensus change(s)")
    for c in report["asset_changes"]:
        risk_tag = f"  [{c['transition_risk']} risk]" if c.get("transition_risk") else ""
        print(f"  {c['ticker']}: {c['from_regime']} → {c['to_regime']}{risk_tag}")
    if report["consensus_change"]:
        cc = report["consensus_change"]
        print(f"  consensus: {cc['from_regime']} ({cc['from_level']}) → {cc['to_regime']} ({cc['to_level']})")

    if args.dry_run:
        resend_config = None
        smtp_config = None
    else:
        resend_config = _build_resend_config()
        smtp_config = None if resend_config else _build_smtp_config()
        if not (resend_config or smtp_config):
            print("[alerts] no email backend (set RESEND_API_KEY or ALERT_SMTP_*) — "
                  "email delivery disabled (webhooks still fire)")

    result = dispatch_alerts(
        report,
        subscriber_store_path=STORE_PATH,
        smtp_config=smtp_config,
        resend_config=resend_config,
        dry_run=args.dry_run,
    )

    print(
        f"[alerts] dispatch complete: sent={result['sent']}  "
        f"skipped={result['skipped']}  errors={len(result['errors'])}  "
        f"backend={result.get('backend','?')}"
    )
    for err in result["errors"]:
        print(f"  ! {err}", file=sys.stderr)

    return 1 if result["errors"] else 0


def cmd_add_email(args: argparse.Namespace) -> int:
    sub = add_subscriber(email=args.add_email, path=STORE_PATH)
    print(f"[alerts] subscribed: {sub['email']}")
    print(f"[alerts] total active subscribers: {count_active_subscribers(STORE_PATH)}")
    return 0


def cmd_add_webhook(args: argparse.Namespace) -> int:
    sub = add_subscriber(webhook_url=args.add_webhook, path=STORE_PATH)
    print(f"[alerts] webhook subscribed: {sub['webhook_url'][:60]}...")
    print(f"[alerts] total active subscribers: {count_active_subscribers(STORE_PATH)}")
    return 0


def cmd_remove_email(args: argparse.Namespace) -> int:
    found = remove_subscriber(args.remove_email, path=STORE_PATH)
    if found:
        print(f"[alerts] unsubscribed: {args.remove_email}")
    else:
        print(f"[alerts] not found: {args.remove_email}", file=sys.stderr)
    return 0 if found else 1


def cmd_list(args: argparse.Namespace) -> int:
    subs = load_subscribers(STORE_PATH)
    if not subs:
        print("[alerts] no active subscribers")
        return 0
    print(f"[alerts] {len(subs)} active subscriber(s):")
    for s in subs:
        email_part = f"email={s['email']}" if s.get("email") else ""
        webhook_part = f"webhook={s['webhook_url'][:40]}..." if s.get("webhook_url") else ""
        tickers = ",".join(s.get("tickers") or ["*"])
        parts = [p for p in [email_part, webhook_part] if p]
        print(f"  {' | '.join(parts)}  tickers={tickers}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Regime change alert dispatcher")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be sent without actually sending")
    p.add_argument("--add-email", metavar="EMAIL",
                   help="Add an email subscriber")
    p.add_argument("--add-webhook", metavar="URL",
                   help="Add a webhook subscriber")
    p.add_argument("--remove-email", metavar="EMAIL",
                   help="Unsubscribe an email address")
    p.add_argument("--list", action="store_true",
                   help="List active subscribers")
    args = p.parse_args()

    if args.add_email:
        return cmd_add_email(args)
    if args.add_webhook:
        return cmd_add_webhook(args)
    if args.remove_email:
        return cmd_remove_email(args)
    if args.list:
        return cmd_list(args)
    return cmd_send(args)


if __name__ == "__main__":
    sys.exit(main())
