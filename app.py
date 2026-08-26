from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vixsnap.backtest import (
    EXECUTION_CLOSE_TO_CLOSE,
    EXECUTION_OPEN_TO_CLOSE,
    EXECUTION_OPEN_TO_OPEN,
    execution_description,
    run_backtest,
    threshold_surface,
    yearly_summary,
)
from vixsnap.data import DEFAULT_START, load_market_data
from vixsnap.signals import CASH, LONG_SVXY, LONG_VIXY, StrategyConfig, add_signals, signal_label


st.set_page_config(page_title="VixSnap", page_icon="⚡", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(start: str) -> pd.DataFrame:
    return load_market_data(start=start)


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.2%}"


def number(x: float, digits: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x:.{digits}f}"


def timing_results(
    market: pd.DataFrame,
    config: StrategyConfig,
    cost_bps: float,
) -> tuple:
    research = run_backtest(
        market,
        config,
        cost_bps=cost_bps,
        execution=EXECUTION_CLOSE_TO_CLOSE,
    )
    executable = run_backtest(
        market,
        config,
        cost_bps=cost_bps,
        execution=EXECUTION_OPEN_TO_OPEN,
    )
    return research, executable


def timing_comparison_frame(research, executable) -> pd.DataFrame:
    rows = []
    for label, result in (
        (EXECUTION_CLOSE_TO_CLOSE, research),
        (EXECUTION_OPEN_TO_OPEN, executable),
    ):
        rows.append(
            {
                "Timing": label,
                "Total return": result.metrics["total_return"],
                "CAGR": result.metrics["cagr"],
                "Sharpe": result.metrics["sharpe"],
                "Ann. vol": result.metrics["ann_vol"],
                "Max drawdown": result.metrics["max_drawdown"],
                "Cash exposure": result.exposure["cash_exposure"],
            }
        )
    return pd.DataFrame(rows).set_index("Timing")


def render_timing_comparison(research, executable) -> None:
    st.subheader("Research vs executable timing")
    st.caption(
        "Close-to-close preserves the original VixSnap research test. Next-open → next-open asks what the same "
        "official EOD signal looks like under an executable timing convention."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Research close → close**")
        c1, c2, c3 = st.columns(3)
        c1.metric("CAGR", pct(research.metrics["cagr"]))
        c2.metric("Sharpe", number(research.metrics["sharpe"], 3))
        c3.metric("Max DD", pct(research.metrics["max_drawdown"]))
    with right:
        st.markdown("**Next open → next open**")
        c1, c2, c3 = st.columns(3)
        c1.metric("CAGR", pct(executable.metrics["cagr"]))
        c2.metric("Sharpe", number(executable.metrics["sharpe"], 3))
        c3.metric("Max DD", pct(executable.metrics["max_drawdown"]))

    comparison = timing_comparison_frame(research, executable)
    display = comparison.copy()
    for col in ["Total return", "CAGR", "Ann. vol", "Max drawdown", "Cash exposure"]:
        display[col] = display[col].map(pct)
    display["Sharpe"] = display["Sharpe"].map(lambda x: number(x, 3))
    st.dataframe(display, use_container_width=True)


def latest_signal_panel(signals: pd.DataFrame, config: StrategyConfig) -> None:
    latest = signals.iloc[-1]
    signal = latest["signal"]
    signal_date = signals.index[-1]

    st.subheader("Latest official EOD signal")
    st.caption(
        "The live action panel uses the conservative convention: the final official EOD signal is acted on at the "
        "next market open. Close-to-close remains available separately as a research convention."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Signal", signal_label(signal))
    c2.metric("VIX1D", number(latest["VIX1D"]))
    c3.metric("VIX", number(latest["VIX"]))
    c4.metric("VIX3M", number(latest["VIX3M"]))
    c5.metric("VIX3M − VIX", number(latest["curve_spread"]))

    if signal == LONG_VIXY:
        st.success(
            f"**Next-open action: LONG VIXY.** VIX1D {latest['VIX1D']:.2f} is below "
            f"{config.low_vix1d:g} and the curve condition is satisfied."
        )
    elif signal == LONG_SVXY:
        st.success(
            f"**Next-open action: LONG SVXY.** VIX1D {latest['VIX1D']:.2f} is above "
            f"{config.high_vix1d:g} and VIX ≤ VIX3M."
        )
    else:
        reasons = []
        if config.low_vix1d <= latest["VIX1D"] <= config.high_vix1d:
            reasons.append("VIX1D is inside the neutral band")
        if not latest["curve_ok"]:
            reasons.append("VIX > VIX3M")
        detail = "; ".join(reasons) if reasons else "no entry condition is active"
        st.info(f"**Next-open action: CASH.** {detail}.")

    st.caption(f"Signal date: {signal_date:%Y-%m-%d}")


def overview_page(market: pd.DataFrame, config: StrategyConfig, cost_bps: float) -> None:
    signals = add_signals(market, config)
    latest_signal_panel(signals, config)

    st.divider()
    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("VIX1D regime history")
        recent = signals.tail(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent["VIX1D"], name="VIX1D"))
        fig.add_hline(y=config.low_vix1d, line_dash="dash", annotation_text="VIXY threshold")
        fig.add_hline(y=config.high_vix1d, line_dash="dash", annotation_text="SVXY threshold")
        fig.update_layout(height=420, yaxis_title="Volatility index", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Signal mix")
        counts = signals["signal"].value_counts().reindex([LONG_VIXY, LONG_SVXY, CASH], fill_value=0)
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55)
        fig.update_layout(height=420, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent signal events")
    events = signals.loc[signals["signal"].ne(signals["signal"].shift())].copy().tail(20)
    events = events[["VIX1D", "VIX", "VIX3M", "curve_spread", "signal"]].sort_index(ascending=False)
    st.dataframe(events, use_container_width=True)

    st.divider()
    research, executable = timing_results(market, config, cost_bps)
    render_timing_comparison(research, executable)
    st.caption(f"Both comparisons use {cost_bps:g} bps per one-way 100% notional trade.")


def backtest_page(market: pd.DataFrame, config: StrategyConfig, cost_bps: float, execution: str) -> None:
    st.header("Backtest")

    research, executable = timing_results(market, config, cost_bps)
    render_timing_comparison(research, executable)

    comparison_equity = pd.concat(
        [
            research.daily["equity"].rename(EXECUTION_CLOSE_TO_CLOSE),
            executable.daily["equity"].rename(EXECUTION_OPEN_TO_OPEN),
        ],
        axis=1,
    )
    fig = px.line(comparison_equity, title="Timing comparison: growth of $1")
    fig.update_layout(height=430, yaxis_title="Equity", xaxis_title=None, legend_title_text="Timing")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(f"Detailed view — {execution}")
    st.caption(execution_description(execution))

    result = run_backtest(market, config, cost_bps=cost_bps, execution=execution)
    m = result.metrics

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total return", pct(m["total_return"]))
    c2.metric("CAGR", pct(m["cagr"]))
    c3.metric("Sharpe", number(m["sharpe"], 3))
    c4.metric("Ann. vol", pct(m["ann_vol"]))
    c5.metric("Max DD", pct(m["max_drawdown"]))

    fig = px.line(result.daily, y="equity", title=f"Growth of $1 — {execution}")
    fig.update_layout(height=430, yaxis_title="Equity", xaxis_title=None, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.area(result.daily, y="drawdown", title=f"Drawdown — {execution}")
    fig.update_layout(height=300, yaxis_tickformat=".0%", xaxis_title=None, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Exposure")
        exposure = pd.Series(result.exposure, name="value").to_frame()
        st.dataframe(exposure, use_container_width=True)
    with right:
        st.subheader("Leg diagnostics")
        legs = result.leg_stats.copy()
        for col in ["mean_return", "median_return", "hit_rate", "cumulative_gross_return", "best_day", "worst_day"]:
            legs[col] = legs[col].map(pct)
        st.dataframe(legs, use_container_width=True)

    st.subheader("Year-by-year")
    yearly = yearly_summary(result)
    display = yearly.copy()
    display["return"] = display["return"].map(pct)
    display["max_drawdown"] = display["max_drawdown"].map(pct)
    display["sharpe"] = display["sharpe"].map(lambda x: number(x, 3))
    st.dataframe(display, use_container_width=True)

    safe_execution = execution.lower().replace(" ", "_").replace("→", "to")
    st.download_button(
        "Download detailed backtest daily CSV",
        result.daily.to_csv(index_label="date"),
        file_name=f"vixsnap_{safe_execution}_daily.csv",
        mime="text/csv",
    )


def research_page(market: pd.DataFrame, config: StrategyConfig, cost_bps: float, execution: str) -> None:
    st.header("Research")
    st.caption(f"Threshold research is currently using: **{execution}**")
    if execution == EXECUTION_CLOSE_TO_CLOSE:
        st.info(
            "Close-to-close is intentionally the primary signal-research convention. It measures whether the official "
            "EOD state predicts the next ETF close-to-close return; it is not being presented as a literal 4:00 p.m. entry."
        )

    st.subheader("Does the curve filter bind on the low-VIX1D leg?")
    signals = add_signals(market, config)
    low = signals["VIX1D"].lt(config.low_vix1d)
    rejected = low & ~signals["curve_ok"]
    c1, c2, c3 = st.columns(3)
    c1.metric("VIX1D below low threshold", int(low.sum()))
    c2.metric("Also VIX ≤ VIX3M", int((low & signals["curve_ok"]).sum()))
    c3.metric("Rejected by curve", int(rejected.sum()))

    st.subheader("Threshold stability")
    st.caption(
        "This is for robustness, not parameter mining. A broad plateau around the chosen thresholds is more convincing "
        "than one isolated best cell."
    )
    metric = st.radio("Surface metric", ["sharpe", "cagr"], horizontal=True)
    surface = threshold_surface(
        market,
        lows=[8, 9, 10, 11, 12],
        highs=[18, 20, 22, 25, 30],
        cost_bps=cost_bps,
        execution=execution,
        metric=metric,
    )
    fig = px.imshow(
        surface,
        text_auto=".2f" if metric == "sharpe" else ".1%",
        aspect="auto",
        labels={"x": "High VIX1D threshold", "y": "Low VIX1D threshold", "color": metric.upper()},
        title=f"{metric.upper()} threshold surface — {execution}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("VIX1D vs curve spread")
    sample = signals.reset_index().rename(columns={signals.index.name or "index": "date"})
    fig = px.scatter(
        sample,
        x="VIX1D",
        y="curve_spread",
        color="signal",
        hover_data=["date", "VIX", "VIX3M"],
        labels={"curve_spread": "VIX3M − VIX"},
    )
    fig.add_vline(x=config.low_vix1d, line_dash="dash")
    fig.add_vline(x=config.high_vix1d, line_dash="dash")
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("⚡ VixSnap")
    st.caption("VIX1D extreme-reversion signal dashboard")

    with st.sidebar:
        st.header("Strategy")
        low = st.number_input("Long VIXY when VIX1D <", min_value=1.0, max_value=19.0, value=10.0, step=0.5)
        high = st.number_input("Long SVXY when VIX1D >", min_value=11.0, max_value=80.0, value=20.0, step=0.5)
        require_vixy_curve = st.checkbox("Require VIX ≤ VIX3M for VIXY", value=True)
        require_svxy_curve = st.checkbox("Require VIX ≤ VIX3M for SVXY", value=True)
        st.divider()
        execution = st.selectbox(
            "Detailed / research timing",
            [EXECUTION_CLOSE_TO_CLOSE, EXECUTION_OPEN_TO_OPEN, EXECUTION_OPEN_TO_CLOSE],
            help=(
                "Close-to-close preserves the original research test. Next-open → next-open is the executable EOD-signal "
                "implementation. The Backtest page always compares the first two side by side."
            ),
        )
        cost_bps = st.number_input("One-way cost (bps)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        st.divider()
        page = st.radio("Page", ["Overview", "Backtest", "Research"])
        if st.button("Refresh market data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    config = StrategyConfig(
        low_vix1d=float(low),
        high_vix1d=float(high),
        require_curve_for_vixy=require_vixy_curve,
        require_curve_for_svxy=require_svxy_curve,
    )

    try:
        with st.spinner("Loading Cboe indices and ETF prices…"):
            market = get_market_data(DEFAULT_START)
    except Exception as exc:
        st.error("Could not load market data.")
        st.exception(exc)
        st.stop()

    if market.empty:
        st.warning("No aligned market data were returned.")
        st.stop()

    if page == "Overview":
        overview_page(market, config, cost_bps)
    elif page == "Backtest":
        backtest_page(market, config, cost_bps, execution)
    else:
        research_page(market, config, cost_bps, execution)


if __name__ == "__main__":
    main()
