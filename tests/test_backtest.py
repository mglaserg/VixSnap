import pandas as pd

from vixsnap.backtest import (
    EXECUTION_CLOSE_TO_CLOSE,
    EXECUTION_OPEN_TO_OPEN,
    run_backtest,
)
from vixsnap.signals import StrategyConfig


def make_market() -> pd.DataFrame:
    idx = pd.date_range("2026-01-02", periods=4, freq="B")
    return pd.DataFrame(
        {
            "VIX1D": [9.0, 15.0, 15.0, 15.0],
            "VIX": [15.0] * 4,
            "VIX3M": [17.0] * 4,
            "VIXY_open": [100.0, 100.0, 105.0, 105.0],
            "VIXY_close": [100.0, 110.0, 121.0, 121.0],
            "SVXY_open": [100.0] * 4,
            "SVXY_close": [100.0] * 4,
        },
        index=idx,
    )


def test_signal_is_lagged_to_next_open():
    market = make_market()
    idx = market.index
    result = run_backtest(
        market,
        StrategyConfig(),
        cost_bps=0.0,
        execution=EXECUTION_OPEN_TO_OPEN,
    )
    # Signal on day 0 is entered at day 1 open and earns day1->day2 open return.
    assert result.daily.loc[idx[0], "position"] == "CASH"
    assert result.daily.loc[idx[1], "position"] == "VIXY"
    assert round(result.daily.loc[idx[1], "strategy_ret"], 10) == 0.05


def test_close_to_close_reproduces_original_research_convention():
    market = make_market()
    idx = market.index
    result = run_backtest(
        market,
        StrategyConfig(),
        cost_bps=0.0,
        execution=EXECUTION_CLOSE_TO_CLOSE,
    )
    # Day-0 EOD signal is associated with day0-close -> day1-close return.
    # The row carrying that return is day 1 because pct_change is indexed by
    # the ending close. This is intentionally a research convention, not an
    # executable claim about knowing final VIX1D at the 4:00 ETF close.
    assert result.daily.loc[idx[1], "position"] == "VIXY"
    assert round(result.daily.loc[idx[1], "VIXY_trade_ret"], 10) == 0.1
    assert round(result.daily.loc[idx[1], "strategy_ret"], 10) == 0.1


def test_research_and_executable_modes_can_differ():
    market = make_market()
    research = run_backtest(
        market,
        StrategyConfig(),
        cost_bps=0.0,
        execution=EXECUTION_CLOSE_TO_CLOSE,
    )
    executable = run_backtest(
        market,
        StrategyConfig(),
        cost_bps=0.0,
        execution=EXECUTION_OPEN_TO_OPEN,
    )
    assert research.daily["strategy_ret"].tolist() != executable.daily["strategy_ret"].tolist()
