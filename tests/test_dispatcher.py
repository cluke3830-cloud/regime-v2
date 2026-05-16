"""Tests for src/alerts/dispatcher.py (Phase 6).

Subscriber store operations are tested with a tmp_path fixture so nothing
touches the real data/subscribers.json. Delivery (SMTP + webhook) is tested
via monkey-patching — no actual network or SMTP calls are made.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts.dispatcher import (  # noqa: E402
    add_subscriber,
    count_active_subscribers,
    dispatch_alerts,
    load_subscribers,
    remove_subscriber,
    save_subscribers,
    _filter_changes_for_subscriber,
)


# ---------------------------------------------------------------------------
# Subscriber store helpers
# ---------------------------------------------------------------------------


class TestSubscriberStore:
    def test_empty_store_returns_empty_list(self, tmp_path):
        p = tmp_path / "subs.json"
        assert load_subscribers(p) == []

    def test_add_email_subscriber(self, tmp_path):
        p = tmp_path / "subs.json"
        sub = add_subscriber(email="a@example.com", path=p)
        assert sub["email"] == "a@example.com"
        assert sub["active"] is True
        assert load_subscribers(p)[0]["email"] == "a@example.com"

    def test_add_webhook_subscriber(self, tmp_path):
        p = tmp_path / "subs.json"
        sub = add_subscriber(webhook_url="https://hooks.example.com/x", path=p)
        assert sub["webhook_url"] == "https://hooks.example.com/x"

    def test_duplicate_email_updates_not_duplicates(self, tmp_path):
        p = tmp_path / "subs.json"
        add_subscriber(email="a@example.com", path=p)
        add_subscriber(email="a@example.com", webhook_url="https://wh.example.com", path=p)
        subs = load_subscribers(p)
        assert len(subs) == 1
        assert subs[0]["webhook_url"] == "https://wh.example.com"

    def test_remove_subscriber_soft_deletes(self, tmp_path):
        p = tmp_path / "subs.json"
        add_subscriber(email="b@example.com", path=p)
        found = remove_subscriber("b@example.com", path=p)
        assert found is True
        assert load_subscribers(p) == []  # inactive excluded by load

    def test_remove_nonexistent_returns_false(self, tmp_path):
        p = tmp_path / "subs.json"
        assert remove_subscriber("nobody@example.com", path=p) is False

    def test_count_active(self, tmp_path):
        p = tmp_path / "subs.json"
        add_subscriber(email="x@example.com", path=p)
        add_subscriber(email="y@example.com", path=p)
        assert count_active_subscribers(p) == 2
        remove_subscriber("x@example.com", path=p)
        assert count_active_subscribers(p) == 1

    def test_add_without_email_or_webhook_raises(self, tmp_path):
        p = tmp_path / "subs.json"
        with pytest.raises(ValueError):
            add_subscriber(path=p)

    def test_inactive_subscriber_not_returned_by_load(self, tmp_path):
        p = tmp_path / "subs.json"
        subs = [{"email": "z@example.com", "active": False}]
        save_subscribers(subs, p)
        assert load_subscribers(p) == []


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFilterChanges:
    def _report(self, tickers=("SPY",)) -> dict:
        return {
            "has_changes": True,
            "asset_changes": [
                {"ticker": t, "name": t, "from_regime": "Bull", "to_regime": "Bear",
                 "from_label": 0, "to_label": 2, "transition_risk": None}
                for t in tickers
            ],
            "consensus_change": {
                "from_regime": "Bull", "from_level": "strong",
                "to_regime": "Bear", "to_level": "split",
            },
            "generated_at": "2026-05-16T22:00:00Z",
        }

    def test_wildcard_subscriber_sees_all(self):
        sub = {"tickers": ["*"], "notify_consensus": True}
        filtered = _filter_changes_for_subscriber(sub, self._report(["SPY", "TLT"]))
        assert len(filtered["asset_changes"]) == 2
        assert filtered["consensus_change"] is not None

    def test_ticker_specific_subscriber_filtered(self):
        sub = {"tickers": ["SPY"], "notify_consensus": False}
        filtered = _filter_changes_for_subscriber(sub, self._report(["SPY", "TLT"]))
        assert len(filtered["asset_changes"]) == 1
        assert filtered["asset_changes"][0]["ticker"] == "SPY"
        assert filtered["consensus_change"] is None

    def test_no_matching_ticker_no_changes(self):
        sub = {"tickers": ["GLD"], "notify_consensus": False}
        filtered = _filter_changes_for_subscriber(sub, self._report(["SPY"]))
        assert filtered["has_changes"] is False


# ---------------------------------------------------------------------------
# Dispatch (dry-run / monkey-patch)
# ---------------------------------------------------------------------------


class TestDispatch:
    def _report(self) -> dict:
        return {
            "generated_at": "2026-05-16T22:00:00Z",
            "has_changes": True,
            "asset_changes": [
                {"ticker": "SPY", "name": "S&P 500", "from_regime": "Bull",
                 "to_regime": "Bear", "from_label": 0, "to_label": 2,
                 "transition_risk": "high"},
            ],
            "consensus_change": None,
            "no_change_tickers": [],
        }

    def test_dry_run_does_not_call_send_email(self, capsys):
        subs = [{"email": "t@example.com", "tickers": ["*"],
                 "notify_consensus": True, "active": True}]
        result = dispatch_alerts(
            self._report(), subscribers=subs,
            smtp_config={"host": "smtp.example.com", "port": 587,
                         "user": "u", "password": "p"},
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert result["sent"] == 1
        assert result["errors"] == []
        out = capsys.readouterr().out
        assert "dry-run" in out

    def test_no_smtp_config_no_email_sent(self, monkeypatch):
        sent_calls = []
        monkeypatch.setattr(
            "src.alerts.dispatcher._send_email",
            lambda *a, **kw: sent_calls.append(a),
        )
        subs = [{"email": "t@example.com", "tickers": ["*"],
                 "notify_consensus": True, "active": True}]
        result = dispatch_alerts(self._report(), subscribers=subs, smtp_config=None)
        assert sent_calls == []
        assert result["sent"] == 0

    def test_webhook_called_for_webhook_subscriber(self, monkeypatch):
        webhook_calls = []
        monkeypatch.setattr(
            "src.alerts.dispatcher._send_webhook",
            lambda url, payload, **kw: webhook_calls.append(url),
        )
        subs = [{"webhook_url": "https://hook.example.com", "tickers": ["*"],
                 "notify_consensus": True, "active": True}]
        result = dispatch_alerts(self._report(), subscribers=subs)
        assert len(webhook_calls) == 1
        assert result["sent"] == 1

    def test_no_changes_skips_subscriber(self):
        report = {**self._report(), "has_changes": False, "asset_changes": [],
                  "consensus_change": None}
        subs = [{"email": "t@example.com", "tickers": ["*"],
                 "notify_consensus": True, "active": True}]
        result = dispatch_alerts(report, subscribers=subs, smtp_config=None)
        assert result["skipped"] == 1
        assert result["sent"] == 0

    def test_resend_preferred_over_smtp_when_both_set(self, monkeypatch):
        resend_calls, smtp_calls = [], []
        monkeypatch.setattr(
            "src.alerts.dispatcher._send_via_resend",
            lambda *a, **kw: resend_calls.append(kw.get("api_key")),
        )
        monkeypatch.setattr(
            "src.alerts.dispatcher._send_email",
            lambda *a, **kw: smtp_calls.append(a),
        )
        subs = [{"email": "t@example.com", "tickers": ["*"],
                 "notify_consensus": True, "active": True}]
        result = dispatch_alerts(
            self._report(), subscribers=subs,
            resend_config={"api_key": "re_xyz", "from_email": "f@x.com"},
            smtp_config={"host": "h", "port": 587, "user": "u", "password": "p"},
        )
        assert resend_calls == ["re_xyz"]  # Resend used
        assert smtp_calls == []             # SMTP skipped
        assert result["backend"] == "resend"
        assert result["sent"] == 1

    def test_resend_error_recorded_not_raised(self, monkeypatch):
        monkeypatch.setattr(
            "src.alerts.dispatcher._send_via_resend",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("rate limited")),
        )
        subs = [{"email": "t@example.com", "tickers": ["*"],
                 "notify_consensus": True, "active": True}]
        result = dispatch_alerts(
            self._report(), subscribers=subs,
            resend_config={"api_key": "re_xyz", "from_email": "f@x.com"},
        )
        assert result["errors"]
        assert "rate limited" in result["errors"][0]
        assert result["sent"] == 0

    def test_webhook_error_recorded_not_raised(self, monkeypatch):
        monkeypatch.setattr(
            "src.alerts.dispatcher._send_webhook",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network error")),
        )
        subs = [{"webhook_url": "https://bad.example.com", "tickers": ["*"],
                 "notify_consensus": True, "active": True}]
        result = dispatch_alerts(self._report(), subscribers=subs)
        assert result["errors"]
        assert "network error" in result["errors"][0]
        assert result["sent"] == 0
