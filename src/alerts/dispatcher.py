"""Alert dispatcher for regime change events (Phase 6 + 9).

Email delivery — picks the first configured backend in priority order:
  1. Resend     (Phase 9, recommended) — set RESEND_API_KEY
  2. SMTP       (Phase 6, fallback)    — set ALERT_SMTP_HOST/USER/PASS
Webhook delivery — HTTP POST (Slack incoming webhooks, Zapier, n8n)

Subscriber store
----------------
``data/subscribers.json`` — a JSON list of subscriber dicts::

    [
      {
        "email": "trader@example.com",       # optional
        "webhook_url": "https://...",         # optional
        "tickers": ["*"],                     # ["*"] = all assets
        "notify_consensus": true,             # alert on consensus changes too
        "active": true
      }
    ]

SMTP configuration (env vars)
------------------------------
  ALERT_SMTP_HOST   — e.g. smtp.gmail.com
  ALERT_SMTP_PORT   — e.g. 587
  ALERT_SMTP_USER   — Gmail address or "apikey" for SendGrid
  ALERT_SMTP_PASS   — App password or SendGrid API key
  ALERT_FROM_EMAIL  — From address shown in email headers
"""
from __future__ import annotations

import json
import smtplib
import urllib.request
import urllib.error
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

REGIME_EMOJI = {"Bull": "🟢", "Neutral": "⚪", "Bear": "🔴"}
RISK_EMOJI = {"high": "⚠️", "medium": "🔶", "low": ""}

_DEFAULT_STORE = Path(__file__).resolve().parent.parent.parent / "data" / "subscribers.json"


# ---------------------------------------------------------------------------
# Subscriber store helpers
# ---------------------------------------------------------------------------


def load_subscribers(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or _DEFAULT_STORE
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return [s for s in data if s.get("active", True)]
    except Exception:
        return []


def save_subscribers(subscribers: List[Dict[str, Any]], path: Optional[Path] = None) -> None:
    p = path or _DEFAULT_STORE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(subscribers, indent=2))


def add_subscriber(
    email: Optional[str] = None,
    webhook_url: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    notify_consensus: bool = True,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Add a new subscriber. Returns the new record."""
    if not email and not webhook_url:
        raise ValueError("at least one of email or webhook_url is required")

    p = path or _DEFAULT_STORE
    # Load including inactive so we don't duplicate
    raw: List[Dict[str, Any]] = []
    if p.exists():
        try:
            raw = json.loads(p.read_text())
        except Exception:
            raw = []

    # Deduplicate by email
    if email:
        for existing in raw:
            if existing.get("email") == email:
                existing["active"] = True
                if webhook_url:
                    existing["webhook_url"] = webhook_url
                if tickers:
                    existing["tickers"] = tickers
                save_subscribers(raw, p)
                return existing

    new_sub: Dict[str, Any] = {
        "email": email,
        "webhook_url": webhook_url,
        "tickers": tickers or ["*"],
        "notify_consensus": notify_consensus,
        "active": True,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    raw.append(new_sub)
    save_subscribers(raw, p)
    return new_sub


def remove_subscriber(email: str, path: Optional[Path] = None) -> bool:
    """Soft-delete by setting active=False. Returns True if found."""
    p = path or _DEFAULT_STORE
    if not p.exists():
        return False
    try:
        raw: List[Dict[str, Any]] = json.loads(p.read_text())
    except Exception:
        return False
    found = False
    for sub in raw:
        if sub.get("email") == email:
            sub["active"] = False
            found = True
    if found:
        save_subscribers(raw, p)
    return found


def count_active_subscribers(path: Optional[Path] = None) -> int:
    return len(load_subscribers(path))


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def _subscriber_wants(sub: Dict[str, Any], ticker: str) -> bool:
    tickers = sub.get("tickers") or ["*"]
    return "*" in tickers or ticker in tickers


def _filter_changes_for_subscriber(
    sub: Dict[str, Any],
    change_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of change_report filtered to what this subscriber cares about."""
    asset_changes = [
        c for c in change_report.get("asset_changes", [])
        if _subscriber_wants(sub, c["ticker"])
    ]
    consensus_change = (
        change_report.get("consensus_change")
        if sub.get("notify_consensus", True)
        else None
    )
    has_changes = bool(asset_changes) or consensus_change is not None
    return {
        **change_report,
        "asset_changes": asset_changes,
        "consensus_change": consensus_change,
        "has_changes": has_changes,
    }


# ---------------------------------------------------------------------------
# Email formatting
# ---------------------------------------------------------------------------


def _format_subject(report: Dict[str, Any]) -> str:
    changes = report.get("asset_changes", [])
    if not changes and not report.get("consensus_change"):
        return "Regime Monitor — No Changes"
    tickers = ", ".join(c["ticker"] for c in changes[:3])
    suffix = f" +{len(changes) - 3} more" if len(changes) > 3 else ""
    date_str = (report.get("curr_date") or "")[:10]
    return f"Regime Alert: {tickers}{suffix} ({date_str})"


def _format_text_body(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    curr_date = (report.get("curr_date") or "")[:10]
    lines.append(f"Regime Change Alert — {curr_date}")
    lines.append("=" * 50)

    cons = report.get("consensus_change")
    if cons:
        lines.append(
            f"\nMarket Consensus: {cons['from_regime'] or '—'} ({cons['from_level']}) "
            f"→ {cons['to_regime'] or '—'} ({cons['to_level']})"
        )

    changes = report.get("asset_changes", [])
    if changes:
        lines.append("\nRegime Changes:")
        for c in changes:
            from_e = REGIME_EMOJI.get(c["from_regime"], "")
            to_e = REGIME_EMOJI.get(c["to_regime"], "")
            risk = c.get("transition_risk")
            risk_tag = f"  {RISK_EMOJI.get(risk, '')} transition risk: {risk}" if risk and risk != "low" else ""
            lines.append(
                f"  • {c['ticker']} ({c['name']}): "
                f"{from_e} {c['from_regime']} → {to_e} {c['to_regime']}{risk_tag}"
            )
    else:
        lines.append("\nNo individual asset regime changes.")

    lines.append("\n" + "-" * 50)
    lines.append("Regime Monitor | Unsubscribe: reply with UNSUBSCRIBE")
    return "\n".join(lines)


def _format_html_body(report: Dict[str, Any]) -> str:
    curr_date = (report.get("curr_date") or "")[:10]
    rows: List[str] = []
    for c in report.get("asset_changes", []):
        from_e = REGIME_EMOJI.get(c["from_regime"], "")
        to_e = REGIME_EMOJI.get(c["to_regime"], "")
        risk = c.get("transition_risk")
        risk_cell = (
            f'<td style="color:#f59e0b">{RISK_EMOJI.get(risk,"")} {risk}</td>'
            if risk and risk != "low"
            else "<td>—</td>"
        )
        rows.append(
            f"<tr>"
            f"<td><b>{c['ticker']}</b></td>"
            f"<td>{c['name']}</td>"
            f"<td>{from_e} {c['from_regime']}</td>"
            f"<td>→</td>"
            f"<td>{to_e} {c['to_regime']}</td>"
            f"{risk_cell}"
            f"</tr>"
        )
    table = (
        "<table border='0' cellpadding='6' style='border-collapse:collapse'>"
        "<tr><th>Ticker</th><th>Name</th><th>From</th><th></th><th>To</th><th>Risk</th></tr>"
        + "".join(rows)
        + "</table>"
        if rows else "<p>No individual asset regime changes.</p>"
    )
    cons = report.get("consensus_change")
    cons_html = ""
    if cons:
        cons_html = (
            f"<p><b>Market Consensus:</b> "
            f"{cons['from_regime'] or '—'} ({cons['from_level']}) "
            f"→ {cons['to_regime'] or '—'} ({cons['to_level']})</p>"
        )
    return (
        f"<html><body style='font-family:sans-serif'>"
        f"<h2>Regime Alert — {curr_date}</h2>"
        f"{cons_html}"
        f"{table}"
        f"<hr><p style='color:#888;font-size:12px'>Regime Monitor</p>"
        f"</body></html>"
    )


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


def _send_email(
    to: str,
    subject: str,
    text_body: str,
    html_body: str,
    smtp_config: Dict[str, Any],
) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_config.get("from_email", smtp_config["user"])
    msg["To"] = to
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    host = smtp_config["host"]
    port = int(smtp_config.get("port", 587))
    user = smtp_config["user"]
    password = smtp_config["password"]

    with smtplib.SMTP(host, port, timeout=15) as srv:
        srv.ehlo()
        srv.starttls()
        srv.login(user, password)
        srv.sendmail(msg["From"], [to], msg.as_string())


def _send_via_resend(
    to: str,
    subject: str,
    text_body: str,
    html_body: str,
    api_key: str,
    from_email: str,
) -> None:
    """Send via Resend HTTP API (https://resend.com/docs/api-reference/emails).

    Vastly simpler than SMTP — single bearer token, no app passwords,
    free tier covers 3,000 emails/month and 100/day.
    """
    payload = {
        "from": from_email,
        "to": [to],
        "subject": subject,
        "text": text_body,
        "html": html_body,
    }
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "RegimeMonitor/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Resend returned HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"Resend HTTP {e.code}: {body}") from e


def _send_webhook(url: str, payload: Dict[str, Any], timeout: int = 10) -> None:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "RegimeMonitor/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"webhook returned HTTP {resp.status}")


# ---------------------------------------------------------------------------
# Main dispatch entry point
# ---------------------------------------------------------------------------


def dispatch_alerts(
    change_report: Dict[str, Any],
    subscribers: Optional[List[Dict[str, Any]]] = None,
    smtp_config: Optional[Dict[str, Any]] = None,
    resend_config: Optional[Dict[str, Any]] = None,
    subscriber_store_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Send alerts to all matching subscribers.

    Parameters
    ----------
    change_report : dict
        Output of ``detect_regime_changes``.
    subscribers : list, optional
        Explicit subscriber list. Loads from store if None.
    resend_config : dict, optional
        ``{api_key, from_email}`` — recommended path (Phase 9).
    smtp_config : dict, optional
        ``{host, port, user, password, from_email}`` — legacy fallback.
        Resend wins when both are set.
    subscriber_store_path : Path, optional
        Path to subscribers.json. Uses default if None.
    dry_run : bool
        If True, print but do not actually send.

    Returns
    -------
    dict
        ``{sent: int, skipped: int, errors: [str], dry_run: bool, backend: str}``
    """
    if subscribers is None:
        subscribers = load_subscribers(subscriber_store_path)

    backend = (
        "resend" if resend_config
        else "smtp" if smtp_config
        else "none"
    )

    sent = 0
    skipped = 0
    errors: List[str] = []

    for sub in subscribers:
        filtered = _filter_changes_for_subscriber(sub, change_report)
        if not filtered["has_changes"]:
            skipped += 1
            continue

        subject = _format_subject(filtered)
        text_body = _format_text_body(filtered)
        html_body = _format_html_body(filtered)

        # Email delivery — Resend takes priority over SMTP when both configured
        if sub.get("email"):
            if dry_run:
                print(f"[dry-run] email ({backend}) → {sub['email']}  subject: {subject}")
                print(text_body)
                sent += 1
            elif resend_config:
                try:
                    _send_via_resend(
                        sub["email"], subject, text_body, html_body,
                        api_key=resend_config["api_key"],
                        from_email=resend_config.get("from_email", "alerts@resend.dev"),
                    )
                    sent += 1
                except Exception as exc:
                    errors.append(f"resend:{sub['email']}: {exc}")
            elif smtp_config:
                try:
                    _send_email(sub["email"], subject, text_body, html_body, smtp_config)
                    sent += 1
                except Exception as exc:
                    errors.append(f"smtp:{sub['email']}: {exc}")

        # Webhook delivery
        if sub.get("webhook_url"):
            webhook_payload = {
                "source": "regime_monitor",
                "generated_at": change_report.get("generated_at"),
                "report": filtered,
            }
            if dry_run:
                print(f"[dry-run] webhook → {sub['webhook_url'][:60]}...")
                sent += 1
            else:
                try:
                    _send_webhook(sub["webhook_url"], webhook_payload)
                    sent += 1
                except Exception as exc:
                    errors.append(f"webhook:{sub.get('webhook_url','?')[:40]}: {exc}")

    return {
        "sent": sent,
        "skipped": skipped,
        "errors": errors,
        "dry_run": dry_run,
        "backend": backend,
    }


__all__ = [
    "load_subscribers",
    "save_subscribers",
    "add_subscriber",
    "remove_subscriber",
    "count_active_subscribers",
    "dispatch_alerts",
]
