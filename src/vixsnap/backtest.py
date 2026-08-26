from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import CASH, LONG_SVXY, LONG_VIXY, StrategyConfig, add_signals


EXECUTION_CLOSE_TO_CLOSE = "Research close → close"
EXECUTION_OPEN_TO_OPEN = "Next open → next open"
EXECUTION_OPEN_TO_CLOSE = "Next open → same close"


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    metrics: dict[str, float]
    exposure: dict[str, float]
    leg_stats: pd.DataFrame


def _performance_metrics(returns: pd.Series) -> dict[str, float]:
    r = returns.fillna(0.0)
    if r.empty:
        return {}

    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    std = r.std(ddof=1)
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(equity.iloc[-1] ** (252.0 / len(r)) - 1.0),
        "ann_vol": float(std * np.sqrt(252.0)),
        "sharpe": float((r.mean() / std) * np.sqrt(252.0)) if std > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }


def execution_description(execution: str) -> str:
    if execution == EXECUTION_CLOSE_TO_CLOSE:
        return (
            "Research convention: day t's official EOD signal is associated with the ETF close-to-close "
            "return from t to t+1. Because the final VIX1D value is known after the 4:00 p.m. ETF close, "
            "this measures predictive power and is not a literal executable entry assumption."
        )
    if execution == EXECUTION_OPEN_TO_OPEN:
        return (
            "Executable convention: observe day t's official EOD signal, enter at the day t+1 open, "
            "and hold until the day t+2 open."
        )
    if execution == EXECUTION_OPEN_TO_CLOSE:
        return (
            "Intraday diagnostic: observe day t's official EOD signal, enter at the day t+1 open, "
            "and exit at that same day's close."
        )
    raise ValueError(f"Unknown execution mode: {execution}")


def run_backtest(
    market: pd.DataFrame,
    config: StrategyConfig = StrategyConfig(),
    *,
    cost_bps: float = 5.0,
    execution: str = EXECUTION_CLOSE_TO_CLOSE,
) -> BacktestResult:
    """Run a VixSnap backtest under the requested timing convention.

    Signals are always shifted one row: the position shown on day t comes from
    the volatility-index signal observed on day t-1.

    ``Research close → close`` intentionally reproduces the original VixSnap
    research convention. It associates the final day-t EOD volatility signal
    with the day-t close through day-t+1 close ETF return. That is useful for
    measuring predictiveness, but the final VIX1D close is not known at the
    earlier 4:00 p.m. ETF close.

    ``Next open → next open`` is the executable version of the same signal:
    observe the final EOD signal, then trade from the next open to the following
    open.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")

    df = add_signals(market, config)
    df["position"] = df["signal"].shift(1).fillna(CASH)

    if execution == EXECUTION_CLOSE_TO_CLOSE:
        df["VIXY_trade_ret"] = df["VIXY_close"].pct_change()
        df["SVXY_trade_ret"] = df["SVXY_close"].pct_change()
    elif execution == EXECUTION_OPEN_TO_OPEN:
        df["VIXY_trade_ret"] = df["VIXY_open"].shift(-1) / df["VIXY_open"] - 1.0
        df["SVXY_trade_ret"] = df["SVXY_open"].shift(-1) / df["SVXY_open"] - 1.0
    elif execution == EXECUTION_OPEN_TO_CLOSE:
        df["VIXY_trade_ret"] = df["VIXY_close"] / df["VIXY_open"] - 1.0
        df["SVXY_trade_ret"] = df["SVXY_close"] / df["SVXY_open"] - 1.0
    else:
        raise ValueError(f"Unknown execution mode: {execution}")

    df["w_vixy"] = df["position"].eq(LONG_VIXY).astype(float)
    df["w_svxy"] = df["position"].eq(LONG_SVXY).astype(float)

    gross = (
        df["w_vixy"] * df["VIXY_trade_ret"].fillna(0.0)
        + df["w_svxy"] * df["SVXY_trade_ret"].fillna(0.0)
    )

    if execution == EXECUTION_OPEN_TO_CLOSE:
        # This mode exits at every close and re-enters on each qualifying next
        # open, so each invested day is a round trip.
        turnover = 2.0 * (df["w_vixy"] + df["w_svxy"])
    else:
        turnover = df["w_vixy"].diff().abs().fillna(df["w_vixy"].abs())
        turnover += df["w_svxy"].diff().abs().fillna(df["w_svxy"].abs())

    df["turnover"] = turnover
    df["cost"] = turnover * (cost_bps / 10_000.0)
    df["strategy_ret_gross"] = gross
    df["strategy_ret"] = gross - df["cost"]
    df["equity"] = (1.0 + df["strategy_ret"]).cumprod()
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1.0

    metrics = _performance_metrics(df["strategy_ret"])
    invested = df["position"].ne(CASH)
    exposure = {
        "vixy_days": int(df["position"].eq(LONG_VIXY).sum()),
        "svxy_days": int(df["position"].eq(LONG_SVXY).sum()),
        "cash_days": int(df["position"].eq(CASH).sum()),
        "vixy_exposure": float(df["position"].eq(LONG_VIXY).mean()),
        "svxy_exposure": float(df["position"].eq(LONG_SVXY).mean()),
        "cash_exposure": float(df["position"].eq(CASH).mean()),
        "position_changes": int(df["position"].ne(df["position"].shift()).sum() - 1),
        "one_way_turnover": float(turnover.sum()),
        "invested_hit_rate": float((gross[invested] > 0).mean()) if invested.any() else np.nan,
    }

    rows = []
    for leg, ret_col in ((LONG_VIXY, "VIXY_trade_ret"), (LONG_SVXY, "SVXY_trade_ret")):
        r = df.loc[df["position"].eq(leg), ret_col].dropna()
        rows.append(
            {
                "leg": leg,
                "days": len(r),
                "mean_return": r.mean() if len(r) else np.nan,
                "median_return": r.median() if len(r) else np.nan,
                "hit_rate": (r > 0).mean() if len(r) else np.nan,
                "cumulative_gross_return": (1.0 + r).prod() - 1.0 if len(r) else np.nan,
                "best_day": r.max() if len(r) else np.nan,
                "worst_day": r.min() if len(r) else np.nan,
            }
        )
    leg_stats = pd.DataFrame(rows).set_index("leg")
    return BacktestResult(df, metrics, exposure, leg_stats)


def yearly_summary(result: BacktestResult) -> pd.DataFrame:
    df = result.daily.copy()
    rows = []
    for year, group in df.groupby(df.index.year):
        metrics = _performance_metrics(group["strategy_ret"])
        rows.append(
            {
                "year": int(year),
                "return": (1.0 + group["strategy_ret"]).prod() - 1.0,
                "sharpe": metrics.get("sharpe", np.nan),
                "max_drawdown": metrics.get("max_drawdown", np.nan),
                "vixy_days": int(group["position"].eq(LONG_VIXY).sum()),
                "svxy_days": int(group["position"].eq(LONG_SVXY).sum()),
                "cash_days": int(group["position"].eq(CASH).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def threshold_surface(
    market: pd.DataFrame,
    lows: list[float],
    highs: list[float],
    *,
    cost_bps: float = 5.0,
    execution: str = EXECUTION_CLOSE_TO_CLOSE,
    metric: str = "sharpe",
) -> pd.DataFrame:
    values = []
    for low in lows:
        row = []
        for high in highs:
            result = run_backtest(
                market,
                StrategyConfig(low_vix1d=low, high_vix1d=high),
                cost_bps=cost_bps,
                execution=execution,
            )
            row.append(result.metrics.get(metric, np.nan))
        values.append(row)
    return pd.DataFrame(values, index=lows, columns=highs)
