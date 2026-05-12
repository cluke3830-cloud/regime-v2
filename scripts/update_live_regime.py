"""Daily IBKR-powered regime signal updater.

Replaces the GHA yfinance snapshot job for the live-regime path. Intended
to run on EC2 (where IB Gateway is already live) via a systemd timer at
16:15 ET each weekday — 15 min after market close, after the daily bar
has settled.

Flow
----
1. Call ``build_dashboard_data.main(backend="ibkr")`` to pull today's daily
   bars from IB Gateway, run GMM-HMM + rule-baseline, and write JSON files
   to ``dashboard/public/data/regimes/``.
2. If the snapshot succeeds (≥ 70% asset coverage), git commit and push →
   triggers a Vercel rebuild.
3. Log everything to ``logs/live_regime_update.log``.

Exit codes
----------
0   success + pushed
1   build failed (< 70% asset coverage) — previous snapshot stays on main
2   IBKR connection failed
3   git commit/push failed (data written, but not deployed)

Usage
-----
    python scripts/update_live_regime.py [--dry-run] [--tickers SPY QQQ ...]

Options
-------
--dry-run       Write JSON but skip the git commit / push step.
--tickers ...   Override the default 10-asset universe for quick testing.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [update_live_regime] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "live_regime_update.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _git_commit_push() -> int:
    """Stage updated JSON files, commit, and push. Returns 0 on success."""
    data_dir = ROOT / "dashboard" / "public" / "data"
    add_result = subprocess.run(
        ["git", "add", str(data_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        log.error("git add failed: %s", add_result.stderr.strip())
        return 3

    # Check if there is anything to commit
    diff_result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
    )
    if diff_result.returncode == 0:
        log.info("No changes to commit — regime probabilities unchanged.")
        return 0

    commit_result = subprocess.run(
        ["git", "commit", "-m", "data: live regime update from IBKR"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        log.error("git commit failed: %s", commit_result.stderr.strip())
        return 3

    push_result = subprocess.run(
        ["git", "push"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if push_result.returncode != 0:
        log.error("git push failed: %s", push_result.stderr.strip())
        return 3

    log.info("Committed and pushed updated regime JSON.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write JSON but skip git commit / push.",
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Override universe for testing, e.g. --tickers SPY QQQ.",
    )
    args = parser.parse_args()

    # Temporarily override DEFAULT_UNIVERSE if --tickers supplied
    if args.tickers:
        from src.validation import multi_asset as _ma
        _orig = _ma.DEFAULT_UNIVERSE
        _ma.DEFAULT_UNIVERSE = tuple(args.tickers)

    log.info("Starting live regime update (backend=ibkr, dry_run=%s)", args.dry_run)

    # Verify IBKR reachability before spending time on all tickers.
    try:
        from src.features.ibkr_daily import IbkrDailyClient
        with IbkrDailyClient() as client:
            _ = client.fetch_daily_close("SPY", n_bars=5)
        log.info("IBKR connection OK")
    except Exception as exc:
        log.error("IBKR connection failed: %s", exc)
        return 2

    # Build dashboard data via the shared helper (IBKR backend).
    from scripts.build_dashboard_data import main as build_main

    rc = build_main(backend="ibkr")
    if rc != 0:
        log.error("build_dashboard_data failed with exit code %d", rc)
        return 1

    if args.dry_run:
        log.info("--dry-run: skipping git commit / push.")
        return 0

    push_rc = _git_commit_push()
    if push_rc != 0:
        return push_rc

    log.info("Live regime update complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
