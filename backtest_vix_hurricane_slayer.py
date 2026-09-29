#!/usr/bin/env python3
"""
VIX Hurricane Slayer — workbook-matched Cboe backtest
======================================================

This version was rebuilt from the supplied Robot Wealth Hurricane Slayer
workbook plus the accompanying slayer.py helper.

PRIMARY SIGNAL: WORKBOOK METHOD
-------------------------------
1. Download Cboe first- and second-near-term VX index values:
       VXIND1
       VXIND2
   and Cboe VIX3M.

2. Construct the workbook-style constant-maturity VX exposure:
   - target = 30 CALENDAR days to VX expiration
   - use the first two monthly VX expirations strictly AFTER the calculation date
   - when the nearest future is >30 DTE, constrain to 100% nearest future
   - otherwise linearly interpolate to 30 calendar days

3. Apply the workbook's trading-day volatility adjustment:
       annual trading-day convention = 251
       calendar convention           = 365

   For each VX future:
       adj_factor_vx =
           sqrt((30/365) / (NYSE_sessions(expiry, expiry+30)/251))

   For VIX3M:
       adj_factor_vix3m =
           sqrt((93/365) / (NYSE_sessions(date, date+93)/251))

   NYSE session counts are [start, end), matching the supplied workbook tables.

4. Adjust each VX future individually, then weight:
       VX30_adjusted =
           w1 * (VX1 * adj1) +
           w2 * (VX2 * adj2)

       VIX3M_adjusted =
           VIX3M * adj_vix3m

5. Workbook premium:
       premium = ln(VX30_adjusted / VIX3M_adjusted)

   The workbook also keeps:
       premium_raw = VX30_raw / VIX3M_raw

6. Rolling statistics:
       premium_mean = rolling 252-session mean
       premium_sd   = rolling 252-session sample stdev

       zscore =
           (premium - premium_mean) / premium_sd

       zscore_lagged =
           (premium - premium_mean.shift(1)) /
           premium_sd.shift(1)

   zscore_lagged is the closest historical analogue to the LIVE spreadsheet:
   today's premium is standardized against the last completed historical
   mean and standard deviation.

BACKTEST IMPLEMENTATIONS
------------------------
The spreadsheet defines a cheapness/richness SIGNAL, not a mandatory binary
ETF strategy. Therefore this script reports BOTH:

A) SLAYER LONG-ONLY
       zscore_lagged < threshold  -> VIXY
       otherwise                  -> CASH

B) BINARY VIXY/SVXY EXPERIMENT
       zscore_lagged < threshold  -> VIXY
       otherwise                  -> SVXY

The binary version is our implementation experiment; it should not be
attributed to Robot Wealth as the spreadsheet's exact trading rule.

Timing for both:
       EOD signal at t -> next close-to-close ETF return

HELPER SCRIPT DIAGNOSTIC
------------------------
The supplied slayer.py is simpler than the workbook. It calculates:
       helper_premium = VX30_raw - VIX3M_raw
       helper_zscore  = rolling-252 zscore(helper_premium)

This script calculates that alongside the workbook signal and saves a
comparison. It is NOT the primary signal.

DATA-SOURCE CAVEAT
------------------
The supplied workbook historical table was sourced from RW Lab futures data,
while this backtest uses public Cboe VXIND1/VXIND2 index histories. Thus the
FORMULA is workbook-matched, but the historical PRICE SOURCE is not identical.
Small differences are expected.

Dependencies:
    uv add pandas numpy yfinance matplotlib pandas-market-calendars
"""

from __future__ import annotations

import argparse
import calendar
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import pandas_market_calendars as mcal
except ImportError as exc:
    raise SystemExit(
        "Missing pandas-market-calendars.\n"
        "Install with:\n"
        "  uv add pandas-market-calendars"
    ) from exc


TRADING_DAYS_PER_YEAR = 252
WORKBOOK_TRADING_DAYS = 251
CALENDAR_DAYS_PER_YEAR = 365
VX_TARGET_CALENDAR_DAYS = 30
VIX3M_TARGET_CALENDAR_DAYS = 93

CBOE_DAILY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/"
    "daily_prices/{symbol}_History.csv"
)

NYSE = mcal.get_calendar("NYSE")


# =============================================================================
# Market data
# =============================================================================

def load_cboe_close(symbol: str) -> pd.Series:
    url = CBOE_DAILY_URL.format(symbol=symbol)
    df = pd.read_csv(url)
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "DATE" not in df.columns:
        raise ValueError(
            f"{symbol}: expected DATE column from Cboe; got {list(df.columns)}"
        )

    value_col = next(
        (
            c for c in ("CLOSE", "VALUE", "SETTLE", "SETTLEMENT")
            if c in df.columns
        ),
        None,
    )

    if value_col is None:
        others = [c for c in df.columns if c != "DATE"]
        if len(others) == 1:
            value_col = others[0]
        else:
            raise ValueError(
                f"{symbol}: could not identify value column: {list(df.columns)}"
            )

    idx = pd.to_datetime(df["DATE"], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")

    out = pd.Series(values.to_numpy(), index=idx, name=symbol)
    out = out[~out.index.isna()].dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]

    if out.empty:
        raise RuntimeError(f"{symbol}: no Cboe history returned.")

    return out


def load_etf_prices(start: str, end: str | None) -> pd.DataFrame:
    symbols = ["VIXY", "SVXY"]

    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise RuntimeError("Yahoo returned no VIXY/SVXY prices.")

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        field = "Adj Close" if "Adj Close" in first_level else "Close"
        px = raw[field].copy()
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        px = raw[[field]].copy()
        px.columns = symbols[: len(px.columns)]

    px = px.reindex(columns=symbols)
    px.index = pd.to_datetime(px.index).tz_localize(None)

    return px.dropna(how="any").sort_index()


# =============================================================================
# NYSE trading-day calendar
# =============================================================================

@lru_cache(maxsize=None)
def _nyse_sessions(start_iso: str, end_iso: str) -> pd.DatetimeIndex:
    """
    NYSE sessions in [start, end).

    The spreadsheet's static calendar tables count trading days from the
    forward start date up to, but not including, the forward end date.
    """
    start = pd.Timestamp(start_iso).normalize()
    end = pd.Timestamp(end_iso).normalize()

    if end <= start:
        return pd.DatetimeIndex([])

    sched = NYSE.schedule(
        start_date=start.date(),
        end_date=(end - pd.Timedelta(days=1)).date(),
    )

    return pd.DatetimeIndex(sched.index).tz_localize(None).normalize()


def nyse_session_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return len(
        _nyse_sessions(
            pd.Timestamp(start).normalize().date().isoformat(),
            pd.Timestamp(end).normalize().date().isoformat(),
        )
    )


def previous_nyse_session(day: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(day).normalize()
    lookback = day - pd.Timedelta(days=10)

    sched = NYSE.schedule(
        start_date=lookback.date(),
        end_date=day.date(),
    )

    sessions = pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
    sessions = sessions[sessions <= day]

    if len(sessions) == 0:
        raise RuntimeError(f"Could not locate NYSE session before {day.date()}")

    return sessions[-1]


def is_nyse_session(day: pd.Timestamp) -> bool:
    day = pd.Timestamp(day).normalize()
    return nyse_session_count(day, day + pd.Timedelta(days=1)) == 1


# =============================================================================
# Monthly VX expiry calendar
# =============================================================================

def third_friday(year: int, month: int) -> pd.Timestamp:
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)

    fridays = [
        d
        for d in cal.itermonthdates(year, month)
        if d.month == month and d.weekday() == calendar.FRIDAY
    ]

    return pd.Timestamp(fridays[2])


def next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def monthly_vx_expiration(year: int, month: int) -> pd.Timestamp:
    """
    Generate the standard monthly VX settlement date.

    Normal rule:
      30 calendar days before the SPX monthly expiration in the following month.

    Holiday handling:
      - if the third Friday is not an NYSE session, use the preceding NYSE session
        as the SPX expiration reference
      - if the resulting VX date is not an NYSE session, move to the preceding
        NYSE session

    This reproduces the standard rule closely. The supplied workbook used a
    static contract calendar; unusual exchange-specific exceptions can therefore
    produce small differences versus that static table.
    """
    ny, nm = next_month(year, month)

    spx_exp = third_friday(ny, nm)

    if not is_nyse_session(spx_exp):
        spx_exp = previous_nyse_session(spx_exp)

    vx_exp = spx_exp - pd.Timedelta(days=30)

    if not is_nyse_session(vx_exp):
        vx_exp = previous_nyse_session(vx_exp)

    return vx_exp.normalize()


def make_vx_expirations(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    expirations = []

    for year in range(start.year - 1, end.year + 2):
        for month in range(1, 13):
            expirations.append(monthly_vx_expiration(year, month))

    return pd.DatetimeIndex(sorted(set(expirations)))


# =============================================================================
# Workbook-matched VX30 construction
# =============================================================================

def workbook_vx_weights(
    calc_date: pd.Timestamp,
    expirations: pd.DatetimeIndex,
    target_days: int = VX_TARGET_CALENDAR_DAYS,
) -> dict:
    """
    Reproduce the workbook's constrained VX30 weighting logic.

    Contracts must expire strictly AFTER calc_date.

    If nearest DTE > target:
        nearest weight = 1
        second weight  = 0
        constrained effective maturity > target

    Otherwise:
        linearly interpolate the first two contracts to target DTE.
    """
    d = pd.Timestamp(calc_date).normalize()
    valid = expirations[expirations > d]

    if len(valid) < 2:
        return {
            "expiry1": pd.NaT,
            "expiry2": pd.NaT,
            "dte1": np.nan,
            "dte2": np.nan,
            "weight1": np.nan,
            "weight2": np.nan,
            "effective_maturity": np.nan,
        }

    expiry1 = valid[0]
    expiry2 = valid[1]

    dte1 = int((expiry1 - d).days)
    dte2 = int((expiry2 - d).days)

    if dte1 > target_days:
        w1 = 1.0
        w2 = 0.0
        effective = float(dte1)
    else:
        denom = dte2 - dte1

        if denom <= 0:
            return {
                "expiry1": expiry1,
                "expiry2": expiry2,
                "dte1": dte1,
                "dte2": dte2,
                "weight1": np.nan,
                "weight2": np.nan,
                "effective_maturity": np.nan,
            }

        w2 = (target_days - dte1) / denom
        w1 = 1.0 - w2

        # Spreadsheet constrained weights are nonnegative.
        w1 = float(np.clip(w1, 0.0, 1.0))
        w2 = float(np.clip(w2, 0.0, 1.0))

        total = w1 + w2
        if total > 0:
            w1 /= total
            w2 /= total

        effective = w1 * dte1 + w2 * dte2

    return {
        "expiry1": expiry1,
        "expiry2": expiry2,
        "dte1": dte1,
        "dte2": dte2,
        "weight1": w1,
        "weight2": w2,
        "effective_maturity": effective,
    }


def volatility_calendar_adjustment(
    calendar_days: int,
    trading_days: int,
) -> float:
    """
    Workbook convention:

        sqrt(
            (calendar_days / 365)
            /
            (trading_days / 251)
        )
    """
    if trading_days <= 0:
        return np.nan

    return float(
        np.sqrt(
            (calendar_days / CALENDAR_DAYS_PER_YEAR)
            /
            (trading_days / WORKBOOK_TRADING_DAYS)
        )
    )


def vx_contract_adjustment(expiry: pd.Timestamp) -> tuple[int, float]:
    end = expiry + pd.Timedelta(days=VX_TARGET_CALENDAR_DAYS)
    n = nyse_session_count(expiry, end)

    return n, volatility_calendar_adjustment(
        VX_TARGET_CALENDAR_DAYS,
        n,
    )


def vix3m_adjustment(calc_date: pd.Timestamp) -> tuple[int, float]:
    end = calc_date + pd.Timedelta(days=VIX3M_TARGET_CALENDAR_DAYS)
    n = nyse_session_count(calc_date, end)

    return n, volatility_calendar_adjustment(
        VIX3M_TARGET_CALENDAR_DAYS,
        n,
    )


def construct_workbook_signal_inputs(
    vx1: pd.Series,
    vx2: pd.Series,
    vix3m: pd.Series,
) -> pd.DataFrame:
    base = pd.concat(
        [
            vx1.rename("VXIND1"),
            vx2.rename("VXIND2"),
            vix3m.rename("VIX3M_raw"),
        ],
        axis=1,
        join="inner",
    ).dropna().sort_index()

    expirations = make_vx_expirations(
        base.index.min(),
        base.index.max(),
    )

    rows = []

    # Cache contract adjustment because every contract reuses the same value.
    vx_adj_cache: dict[pd.Timestamp, tuple[int, float]] = {}

    for date, row in base.iterrows():
        weights = workbook_vx_weights(date, expirations)

        if pd.isna(weights["weight1"]):
            continue

        expiry1 = weights["expiry1"]
        expiry2 = weights["expiry2"]

        if expiry1 not in vx_adj_cache:
            vx_adj_cache[expiry1] = vx_contract_adjustment(expiry1)

        if expiry2 not in vx_adj_cache:
            vx_adj_cache[expiry2] = vx_contract_adjustment(expiry2)

        vx1_days, vx1_adj = vx_adj_cache[expiry1]
        vx2_days, vx2_adj = vx_adj_cache[expiry2]

        vix3m_days, vix3m_adj = vix3m_adjustment(date)

        w1 = float(weights["weight1"])
        w2 = float(weights["weight2"])

        p1 = float(row["VXIND1"])
        p2 = float(row["VXIND2"])
        v3 = float(row["VIX3M_raw"])

        vx30_raw = w1 * p1 + w2 * p2

        # Historical workbook methodology:
        # adjust EACH contract, then weight the adjusted prices.
        adjprice1 = p1 * vx1_adj
        adjprice2 = p2 * vx2_adj

        vx30_adjusted = (
            w1 * adjprice1
            + w2 * adjprice2
        )

        vix3m_adjusted = v3 * vix3m_adj

        if (
            vx30_raw <= 0
            or v3 <= 0
            or vx30_adjusted <= 0
            or vix3m_adjusted <= 0
        ):
            continue

        rows.append(
            {
                "date": date,
                "expiry_vx1": expiry1,
                "expiry_vx2": expiry2,
                "tte_vx1": weights["dte1"],
                "tte_vx2": weights["dte2"],
                "weight_vx1": w1,
                "weight_vx2": w2,
                "constrained_effective_maturity": weights["effective_maturity"],
                "price_vx1": p1,
                "price_vx2": p2,
                "calendaradj_vx1": vx1_adj,
                "calendaradj_vx2": vx2_adj,
                "nyse_days_vx1": vx1_days,
                "nyse_days_vx2": vx2_days,
                "adjprice_vx1": adjprice1,
                "adjprice_vx2": adjprice2,
                "vx30_raw": vx30_raw,
                "vx30_adjusted": vx30_adjusted,
                "vix3m_raw": v3,
                "calendaradj_vix3m": vix3m_adj,
                "nyse_days_vix3m": vix3m_days,
                "vix3m_adjusted": vix3m_adjusted,
                # Workbook historical fields:
                "premium_raw": vx30_raw / v3,
                "premium": np.log(vx30_adjusted / vix3m_adjusted),
                # Supplied helper script's simpler definition:
                "helper_premium_difference": vx30_raw - v3,
            }
        )

    if not rows:
        raise RuntimeError("Could not construct any workbook-style VX30 observations.")

    return pd.DataFrame(rows).set_index("date").sort_index()


# =============================================================================
# Z-scores
# =============================================================================

def add_signal_statistics(
    df: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    out = df.copy()

    # Workbook historical signal.
    out["premium_mean"] = (
        out["premium"]
        .rolling(lookback, min_periods=lookback)
        .mean()
    )

    out["premium_sd"] = (
        out["premium"]
        .rolling(lookback, min_periods=lookback)
        .std(ddof=1)
    )

    out["zscore"] = (
        (out["premium"] - out["premium_mean"])
        / out["premium_sd"]
    )

    # Workbook LIVE behavior:
    # today's premium vs the most recently completed rolling history.
    out["premium_mean_lagged"] = out["premium_mean"].shift(1)
    out["premium_sd_lagged"] = out["premium_sd"].shift(1)

    out["zscore_lagged"] = (
        (out["premium"] - out["premium_mean_lagged"])
        / out["premium_sd_lagged"]
    )

    # slayer.py helper diagnostic.
    out["helper_mean"] = (
        out["helper_premium_difference"]
        .rolling(lookback, min_periods=lookback)
        .mean()
    )

    out["helper_sd"] = (
        out["helper_premium_difference"]
        .rolling(lookback, min_periods=lookback)
        .std(ddof=1)
    )

    out["helper_zscore"] = (
        (
            out["helper_premium_difference"]
            - out["helper_mean"]
        )
        / out["helper_sd"]
    )

    return out


# =============================================================================
# Backtesting
# =============================================================================

def max_drawdown(nav: pd.Series) -> float:
    return float(
        (nav / nav.cummax() - 1.0).min()
    )


def performance_metrics(
    returns: pd.Series,
    label: str,
) -> dict:
    r = returns.dropna()

    if len(r) < 2:
        return {
            "Strategy": label,
            "N": len(r),
        }

    nav = (1.0 + r).cumprod()
    sd = r.std(ddof=1)
    years = len(r) / TRADING_DAYS_PER_YEAR

    return {
        "Strategy": label,
        "N": len(r),
        "Total Return": nav.iloc[-1] - 1.0,
        "CAGR": (
            nav.iloc[-1] ** (1.0 / years) - 1.0
            if years > 0 and nav.iloc[-1] > 0
            else np.nan
        ),
        "Ann Vol": sd * np.sqrt(TRADING_DAYS_PER_YEAR),
        "Sharpe": (
            r.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR)
            if sd > 0
            else np.nan
        ),
        "Max Drawdown": max_drawdown(nav),
        "Best Day": r.max(),
        "Worst Day": r.min(),
    }


def run_strategy(
    frame: pd.DataFrame,
    *,
    threshold: float,
    cost_bps: float,
    mode: str,
    z_col: str = "zscore_lagged",
) -> tuple[pd.DataFrame, dict]:
    """
    mode:
        long_only  -> VIXY below threshold, otherwise CASH
        binary     -> VIXY below threshold, otherwise SVXY
    """
    if mode not in {"long_only", "binary"}:
        raise ValueError(mode)

    out = frame.copy()

    if mode == "long_only":
        out["signal"] = np.where(
            out[z_col] < threshold,
            "VIXY",
            "CASH",
        )
    else:
        out["signal"] = np.where(
            out[z_col] < threshold,
            "VIXY",
            "SVXY",
        )

    # EOD signal t earns next close-to-close return, appearing on row t+1.
    out["position"] = out["signal"].shift(1)

    out["w_vixy"] = (
        out["position"].eq("VIXY").astype(float)
    )

    out["w_svxy"] = (
        out["position"].eq("SVXY").astype(float)
    )

    out["strategy_ret_gross"] = (
        out["w_vixy"] * out["VIXY_ret"]
        + out["w_svxy"] * out["SVXY_ret"]
    )

    turnover = (
        out["w_vixy"].diff().abs().fillna(out["w_vixy"].abs())
        + out["w_svxy"].diff().abs().fillna(out["w_svxy"].abs())
    )

    out["turnover"] = turnover
    out["cost"] = turnover * (cost_bps / 10_000.0)

    out["strategy_ret"] = (
        out["strategy_ret_gross"] - out["cost"]
    )

    out["equity"] = (
        1.0 + out["strategy_ret"].fillna(0.0)
    ).cumprod()

    live = out.dropna(subset=["position"]).copy()

    label = (
        f"Slayer long-only z<{threshold:g}"
        if mode == "long_only"
        else f"Binary VIXY/SVXY z<{threshold:g}"
    )

    m = performance_metrics(
        live["strategy_ret"],
        label,
    )

    m.update(
        {
            "Mode": mode,
            "Threshold": threshold,
            "Z-score field": z_col,
            "VIXY days": int(live["position"].eq("VIXY").sum()),
            "SVXY days": int(live["position"].eq("SVXY").sum()),
            "Cash days": int(live["position"].eq("CASH").sum()),
            "VIXY exposure": float(live["position"].eq("VIXY").mean()),
            "Switches": int(
                live["position"]
                .ne(live["position"].shift())
                .sum() - 1
            ),
            "Turnover": float(live["turnover"].sum()),
            "Trading-cost drag": float(live["cost"].sum()),
        }
    )

    return out, m


def state_diagnostics(
    run: pd.DataFrame,
    z_col: str = "zscore_lagged",
) -> pd.DataFrame:
    x = run.dropna(
        subset=["position", "strategy_ret_gross"]
    ).copy()

    rows = []

    for state, g in x.groupby("position"):
        r = g["strategy_ret_gross"]

        rows.append(
            {
                "State": state,
                "Days": len(g),
                "Mean daily return": r.mean(),
                "Median daily return": r.median(),
                "Hit rate": (r > 0).mean(),
                "Cumulative gross return": (
                    (1.0 + r).prod() - 1.0
                ),
                "Mean signal z": g[z_col].mean(),
                "Median signal z": g[z_col].median(),
            }
        )

    return (
        pd.DataFrame(rows).set_index("State")
        if rows
        else pd.DataFrame()
    )


def yearly_metrics(run: pd.DataFrame) -> pd.DataFrame:
    x = run.dropna(
        subset=["position", "strategy_ret"]
    ).copy()

    rows = []

    for year, g in x.groupby(x.index.year):
        m = performance_metrics(
            g["strategy_ret"],
            str(year),
        )

        rows.append(
            {
                "Year": year,
                "Return": (
                    (1.0 + g["strategy_ret"]).prod() - 1.0
                ),
                "Sharpe": m.get("Sharpe"),
                "Max Drawdown": m.get("Max Drawdown"),
                "VIXY days": int(g["position"].eq("VIXY").sum()),
                "SVXY days": int(g["position"].eq("SVXY").sum()),
                "Cash days": int(g["position"].eq("CASH").sum()),
            }
        )

    return (
        pd.DataFrame(rows).set_index("Year")
        if rows
        else pd.DataFrame()
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end", default=None)

    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--threshold", type=float, default=-1.5)
    p.add_argument("--cost-bps", type=float, default=5.0)

    p.add_argument(
        "--outdir",
        default="output/hurricane_slayer_workbook",
    )

    args = p.parse_args()

    print("Downloading Cboe VXIND1...")
    vx1 = load_cboe_close("VXIND1")

    print("Downloading Cboe VXIND2...")
    vx2 = load_cboe_close("VXIND2")

    print("Downloading Cboe VIX3M...")
    vix3m = load_cboe_close("VIX3M")

    print("Building workbook-style adjusted VX30 / VIX3M signal...")
    signal = construct_workbook_signal_inputs(
        vx1=vx1,
        vx2=vx2,
        vix3m=vix3m,
    )

    signal = add_signal_statistics(
        signal,
        lookback=args.lookback,
    )

    print("Downloading VIXY / SVXY adjusted prices...")
    etfs = load_etf_prices(
        start=args.start,
        end=args.end,
    )

    frame = pd.concat(
        [
            signal,
            etfs,
        ],
        axis=1,
        join="inner",
    ).sort_index()

    frame["VIXY_ret"] = frame["VIXY"].pct_change()
    frame["SVXY_ret"] = frame["SVXY"].pct_change()

    frame = frame.loc[pd.Timestamp(args.start):]

    if args.end:
        frame = frame.loc[: pd.Timestamp(args.end)]

    frame = frame.dropna(
        subset=[
            "zscore_lagged",
            "VIXY_ret",
            "SVXY_ret",
        ]
    ).copy()

    if frame.empty:
        raise RuntimeError(
            "No observations remain after Cboe alignment and 252-session warmup."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save the signal table first: this is the main research artifact.
    frame.to_csv(
        outdir / "workbook_signal_and_prices.csv"
    )

    # ---------------------------------------------------------------------
    # Primary implementations
    # ---------------------------------------------------------------------

    long_run, long_summary = run_strategy(
        frame,
        threshold=args.threshold,
        cost_bps=args.cost_bps,
        mode="long_only",
        z_col="zscore_lagged",
    )

    binary_run, binary_summary = run_strategy(
        frame,
        threshold=args.threshold,
        cost_bps=args.cost_bps,
        mode="binary",
        z_col="zscore_lagged",
    )

    long_run.to_csv(
        outdir / "long_only_daily.csv"
    )

    binary_run.to_csv(
        outdir / "binary_daily.csv"
    )

    summaries = pd.DataFrame(
        [long_summary, binary_summary]
    ).set_index("Strategy")

    summaries.to_csv(
        outdir / "summary.csv"
    )

    yearly_metrics(long_run).to_csv(
        outdir / "long_only_yearly.csv"
    )

    yearly_metrics(binary_run).to_csv(
        outdir / "binary_yearly.csv"
    )

    state_diagnostics(long_run).to_csv(
        outdir / "long_only_state_diagnostics.csv"
    )

    state_diagnostics(binary_run).to_csv(
        outdir / "binary_state_diagnostics.csv"
    )

    # ---------------------------------------------------------------------
    # Threshold robustness
    # ---------------------------------------------------------------------

    thresholds = [
        -1.0,
        -1.25,
        -1.5,
        -1.75,
        -2.0,
        -2.25,
        -2.5,
    ]

    threshold_rows = []

    for threshold in thresholds:
        for mode in ("long_only", "binary"):
            _, m = run_strategy(
                frame,
                threshold=threshold,
                cost_bps=args.cost_bps,
                mode=mode,
                z_col="zscore_lagged",
            )
            threshold_rows.append(m)

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df.to_csv(
        outdir / "threshold_diagnostic.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Workbook vs slayer.py signal comparison
    # ---------------------------------------------------------------------

    comparison = frame[
        [
            "vx30_raw",
            "vx30_adjusted",
            "vix3m_raw",
            "vix3m_adjusted",
            "premium_raw",
            "premium",
            "zscore",
            "zscore_lagged",
            "helper_premium_difference",
            "helper_zscore",
        ]
    ].copy()

    comparison["workbook_cheap_-1.5"] = (
        comparison["zscore_lagged"] < -1.5
    )

    comparison["helper_cheap_-1.5"] = (
        comparison["helper_zscore"] < -1.5
    )

    comparison["cheap_signal_disagrees"] = (
        comparison["workbook_cheap_-1.5"]
        != comparison["helper_cheap_-1.5"]
    )

    comparison.to_csv(
        outdir / "workbook_vs_helper_signal.csv"
    )

    signal_corr = comparison[
        ["zscore_lagged", "helper_zscore"]
    ].corr().iloc[0, 1]

    disagreement_rate = (
        comparison["cheap_signal_disagrees"].mean()
    )

    # ---------------------------------------------------------------------
    # Benchmarks
    # ---------------------------------------------------------------------

    benchmarks = []

    for label, returns in [
        ("Always SVXY", frame["SVXY_ret"]),
        ("Always VIXY", frame["VIXY_ret"]),
    ]:
        benchmarks.append(
            performance_metrics(
                returns,
                label,
            )
        )

    benchmark_df = pd.DataFrame(
        benchmarks
    ).set_index("Strategy")

    benchmark_df.to_csv(
        outdir / "benchmarks.csv"
    )

    # ---------------------------------------------------------------------
    # Console formatting
    # ---------------------------------------------------------------------

    def pct(x) -> str:
        return (
            f"{float(x):.2%}"
            if pd.notna(x)
            else "n/a"
        )

    def dec(x) -> str:
        return (
            f"{float(x):.3f}"
            if pd.notna(x)
            else "n/a"
        )

    print("\n" + "=" * 82)
    print("VIX HURRICANE SLAYER — WORKBOOK-MATCHED SIGNAL / CBOE PRICE DATA")
    print("=" * 82)

    print(
        f"Sample after warmup : "
        f"{frame.index.min().date()} -> {frame.index.max().date()}"
    )

    print(f"Observations        : {len(frame):,}")
    print(f"Lookback            : {args.lookback}")
    print(f"Primary signal      : zscore_lagged")
    print(f"Threshold           : {args.threshold:g}")

    latest = frame.iloc[-1]

    print("\n=== LATEST SIGNAL ===")
    print(f"Date                         {frame.index[-1].date()}")
    print(f"VX30 raw                     {latest['vx30_raw']:.4f}")
    print(f"VX30 adjusted                {latest['vx30_adjusted']:.4f}")
    print(f"VIX3M raw                    {latest['vix3m_raw']:.4f}")
    print(f"VIX3M adjusted               {latest['vix3m_adjusted']:.4f}")
    print(f"Premium ln(adj VX30/VIX3M)   {latest['premium']:.6f}")
    print(f"Workbook z-score             {latest['zscore']:+.3f}")
    print(f"Workbook lagged z-score      {latest['zscore_lagged']:+.3f}")
    print(f"slayer.py helper z-score     {latest['helper_zscore']:+.3f}")

    print("\n=== BACKTEST SUMMARY ===")

    pretty = summaries.copy()

    for c in [
        "Total Return",
        "CAGR",
        "Ann Vol",
        "Max Drawdown",
        "Best Day",
        "Worst Day",
        "VIXY exposure",
        "Trading-cost drag",
    ]:
        if c in pretty.columns:
            pretty[c] = pretty[c].map(pct)

    if "Sharpe" in pretty.columns:
        pretty["Sharpe"] = pretty["Sharpe"].map(dec)

    print(
        pretty[
            [
                "Total Return",
                "CAGR",
                "Ann Vol",
                "Sharpe",
                "Max Drawdown",
                "VIXY exposure",
                "VIXY days",
                "SVXY days",
                "Cash days",
                "Switches",
            ]
        ].to_string()
    )

    print("\n=== THRESHOLD ROBUSTNESS ===")

    pt = threshold_df[
        [
            "Mode",
            "Threshold",
            "CAGR",
            "Sharpe",
            "Max Drawdown",
            "VIXY exposure",
            "Switches",
        ]
    ].copy()

    for c in [
        "CAGR",
        "Max Drawdown",
        "VIXY exposure",
    ]:
        pt[c] = pt[c].map(pct)

    pt["Sharpe"] = pt["Sharpe"].map(dec)

    print(pt.to_string(index=False))

    print("\n=== WORKBOOK vs slayer.py HELPER ===")
    print(f"Z-score correlation          {signal_corr:.3f}")
    print(
        "Cheap-signal disagreement   "
        f"{disagreement_rate:.2%} "
        "(threshold -1.5)"
    )

    print("\n=== BENCHMARKS ===")

    pb = benchmark_df.copy()

    for c in [
        "Total Return",
        "CAGR",
        "Ann Vol",
        "Max Drawdown",
        "Best Day",
        "Worst Day",
    ]:
        pb[c] = pb[c].map(pct)

    pb["Sharpe"] = pb["Sharpe"].map(dec)

    print(pb.to_string())

    # ---------------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(11, 6))
        plt.plot(
            long_run.index,
            long_run["equity"],
            label="Slayer trigger: VIXY / cash",
        )
        plt.plot(
            binary_run.index,
            binary_run["equity"],
            label="Binary experiment: VIXY / SVXY",
        )
        plt.yscale("log")
        plt.title("Hurricane Slayer — Workbook-Matched Signal")
        plt.ylabel("Growth of $1 (log scale)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            outdir / "equity.png",
            dpi=160,
        )
        plt.close()

        plt.figure(figsize=(11, 5))
        plt.plot(
            frame.index,
            frame["zscore_lagged"],
            label="Workbook lagged z-score",
        )
        plt.plot(
            frame.index,
            frame["helper_zscore"],
            label="slayer.py helper z-score",
            alpha=0.55,
        )
        plt.axhline(
            args.threshold,
            linestyle="--",
            label=f"Threshold {args.threshold:g}",
        )
        plt.axhline(
            0.0,
            linestyle=":",
        )
        plt.title("Hurricane Slayer Signal Comparison")
        plt.ylabel("Z-score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            outdir / "signal_comparison.png",
            dpi=160,
        )
        plt.close()

        plt.figure(figsize=(11, 5))
        plt.plot(
            frame.index,
            frame["constrained_effective_maturity"],
        )
        plt.axhline(
            30,
            linestyle="--",
        )
        plt.title("VX30 Constrained Effective Maturity")
        plt.ylabel("Calendar days")
        plt.tight_layout()
        plt.savefig(
            outdir / "effective_maturity.png",
            dpi=160,
        )
        plt.close()

    except Exception as exc:
        print(f"\nPlotting skipped: {exc}")

    print(
        "\nIMPORTANT: the workbook defines the Hurricane Slayer cheapness/richness "
        "signal. The VIXY/SVXY binary switch is our backtest implementation, "
        "not a rule asserted by the spreadsheet."
    )

    print(
        "IMPORTANT: signal formulas match the supplied workbook methodology, "
        "but historical VX prices here come from Cboe VXIND1/VXIND2 instead "
        "of the workbook's RW Lab futures history."
    )

    print(
        f"\nSaved results to: {outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
