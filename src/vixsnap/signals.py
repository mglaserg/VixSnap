from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CASH = "CASH"
LONG_VIXY = "VIXY"
LONG_SVXY = "SVXY"


@dataclass(frozen=True)
class StrategyConfig:
    """VixSnap signal thresholds."""

    low_vix1d: float = 10.0
    high_vix1d: float = 20.0
    require_curve_for_vixy: bool = True
    require_curve_for_svxy: bool = True

    def __post_init__(self) -> None:
        if self.low_vix1d >= self.high_vix1d:
            raise ValueError("low_vix1d must be below high_vix1d")


def add_signals(df: pd.DataFrame, config: StrategyConfig = StrategyConfig()) -> pd.DataFrame:
    """Return a copy of *df* with VixSnap curve state and signal columns."""
    required = {"VIX1D", "VIX", "VIX3M"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required signal columns: {sorted(missing)}")

    out = df.copy()
    out["curve_ok"] = out["VIX"] <= out["VIX3M"]
    out["curve_spread"] = out["VIX3M"] - out["VIX"]

    vixy_ok = out["VIX1D"].lt(config.low_vix1d)
    if config.require_curve_for_vixy:
        vixy_ok &= out["curve_ok"]

    svxy_ok = out["VIX1D"].gt(config.high_vix1d)
    if config.require_curve_for_svxy:
        svxy_ok &= out["curve_ok"]

    out["signal"] = CASH
    out.loc[vixy_ok, "signal"] = LONG_VIXY
    out.loc[svxy_ok, "signal"] = LONG_SVXY
    return out


def signal_label(signal: str) -> str:
    return {
        LONG_VIXY: "LONG VIXY",
        LONG_SVXY: "LONG SVXY",
        CASH: "CASH",
    }.get(signal, str(signal))
