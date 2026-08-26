import pandas as pd

from vixsnap.signals import CASH, LONG_SVXY, LONG_VIXY, StrategyConfig, add_signals


def test_signal_rules():
    df = pd.DataFrame(
        {
            "VIX1D": [9.0, 21.0, 21.0, 15.0],
            "VIX": [15.0, 18.0, 22.0, 16.0],
            "VIX3M": [17.0, 20.0, 20.0, 18.0],
        }
    )
    out = add_signals(df, StrategyConfig())
    assert out["signal"].tolist() == [LONG_VIXY, LONG_SVXY, CASH, CASH]


def test_low_threshold_can_ignore_curve():
    df = pd.DataFrame({"VIX1D": [9.0], "VIX": [20.0], "VIX3M": [18.0]})
    out = add_signals(df, StrategyConfig(require_curve_for_vixy=False))
    assert out.loc[0, "signal"] == LONG_VIXY
