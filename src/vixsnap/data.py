from __future__ import annotations

from io import StringIO
import json
from typing import Iterable
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf


CBOE_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
CBOE_QUOTE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{symbol}.json"
DEFAULT_START = "2022-05-13"


def _read_url_csv(url: str) -> pd.DataFrame:
    # A browser-like UA avoids occasional CDN rejection of bare urllib requests.
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 VixSnap/0.1"})
    with urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8")
    return pd.read_csv(StringIO(text))


def _read_url_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 VixSnap/0.1"})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def load_cboe_quote(symbol: str) -> dict[str, object]:
    """Load one current Cboe delayed index quote."""
    symbol = symbol.upper()
    payload = _read_url_json(CBOE_QUOTE_URL.format(symbol=symbol))
    data = payload.get("data", {})
    if "current_price" not in data:
        raise ValueError(f"Unexpected Cboe quote payload for {symbol}")

    return {
        "symbol": symbol,
        "price": float(data["current_price"]),
        "last_trade_time": pd.to_datetime(data.get("last_trade_time"), errors="coerce"),
        "timestamp": pd.to_datetime(payload.get("timestamp"), errors="coerce"),
    }


def load_live_indices() -> pd.DataFrame:
    """Load current delayed VIX1D, VIX, and VIX3M values from Cboe."""
    row: dict[str, object] = {}
    for symbol in ("VIX1D", "VIX", "VIX3M"):
        quote = load_cboe_quote(symbol)
        row[symbol] = quote["price"]
        row[f"{symbol}_time"] = quote["last_trade_time"]
    return pd.DataFrame([row])


def load_cboe_index(symbol: str) -> pd.Series:
    """Load official Cboe daily closes for one volatility index."""
    df = _read_url_csv(CBOE_HISTORY_URL.format(symbol=symbol.upper()))
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "DATE" not in df.columns or "CLOSE" not in df.columns:
        raise ValueError(f"Unexpected Cboe columns for {symbol}: {list(df.columns)}")

    series = pd.Series(
        pd.to_numeric(df["CLOSE"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["DATE"], errors="coerce"),
        name=symbol.upper(),
    )
    series = series[~series.index.isna()].dropna().sort_index()
    series.index = series.index.tz_localize(None)
    return series


def _normalize_yf_download(raw: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    symbols = list(symbols)
    frames: list[pd.DataFrame] = []

    for symbol in symbols:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(-1):
                continue
            part = raw.xs(symbol, axis=1, level=-1).copy()
        else:
            if len(symbols) != 1:
                raise ValueError("Expected MultiIndex columns for multiple Yahoo symbols")
            part = raw.copy()

        wanted = {}
        for field in ("Open", "Close"):
            if field in part.columns:
                wanted[field] = f"{symbol}_{field.lower()}"
        if len(wanted) != 2:
            raise ValueError(f"Yahoo data for {symbol} did not contain Open and Close")

        part = part[list(wanted)].rename(columns=wanted)
        frames.append(part)

    if not frames:
        raise ValueError("Yahoo Finance returned no usable ETF data")

    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def load_etf_prices(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Load split/dividend-adjusted VIXY and SVXY open/close data."""
    symbols = ["VIXY", "SVXY"]
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no VIXY/SVXY data")
    return _normalize_yf_download(raw, symbols)


def load_market_data(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Load and align the three Cboe indices with tradeable ETF prices."""
    indices = pd.concat(
        [load_cboe_index("VIX1D"), load_cboe_index("VIX"), load_cboe_index("VIX3M")],
        axis=1,
        join="inner",
    )
    etfs = load_etf_prices(start=start, end=end)
    df = indices.join(etfs, how="inner").sort_index()
    df = df.loc[pd.Timestamp(start) :]
    if end:
        df = df.loc[: pd.Timestamp(end)]
    return df.dropna()
