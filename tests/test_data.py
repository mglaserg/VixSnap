import pandas as pd

from vixsnap import data


def test_load_cboe_quote_parses_current_price(monkeypatch):
    payload = {
        "timestamp": "2026-08-27 16:05:32",
        "data": {
            "current_price": 8.84,
            "last_trade_time": "2026-08-27T11:50:31",
        },
    }
    monkeypatch.setattr(data, "_read_url_json", lambda url: payload)

    quote = data.load_cboe_quote("vix1d")

    assert quote["symbol"] == "VIX1D"
    assert quote["price"] == 8.84
    assert quote["last_trade_time"] == pd.Timestamp("2026-08-27 11:50:31")


def test_load_live_indices_uses_three_quotes(monkeypatch):
    prices = {"VIX1D": 8.84, "VIX": 14.48, "VIX3M": 17.58}

    def fake_quote(symbol):
        return {
            "symbol": symbol,
            "price": prices[symbol],
            "last_trade_time": pd.Timestamp("2026-08-27 11:50:31"),
            "timestamp": pd.Timestamp("2026-08-27 12:05:32"),
        }

    monkeypatch.setattr(data, "load_cboe_quote", fake_quote)
    live = data.load_live_indices()

    assert live.loc[0, "VIX1D"] == 8.84
    assert live.loc[0, "VIX"] == 14.48
    assert live.loc[0, "VIX3M"] == 17.58
