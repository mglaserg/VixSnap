# VixSnap ⚡

VixSnap is a Streamlit dashboard for researching and tracking a simple VIX1D extreme-reversion strategy.

## Strategy

Using official Cboe end-of-day index values:

- **Long VIXY** when `VIX1D < 10` and `VIX <= VIX3M`.
- **Long SVXY** when `VIX1D > 20` and `VIX <= VIX3M`.
- **Cash** otherwise.

The thresholds and curve filters are configurable in the dashboard.

## Two timing views

VixSnap deliberately keeps **research timing** separate from **executable timing**.

### Research close → close

This preserves the convention used in the original VixSnap study:

1. Classify day `t` from its official EOD VIX1D/VIX/VIX3M state.
2. Associate that state with the ETF return from the day-`t` close to the day-`t+1` close.
3. Use the result to measure whether the volatility state predicts the next close-to-close ETF return.

This is a **research convention**, not a claim that the final VIX1D EOD value was known at the earlier 4:00 p.m. ETF close. The dashboard labels it accordingly.

### Next open → next open

This is the primary executable EOD-signal convention:

1. Observe the official VIX1D/VIX/VIX3M signal after day `t`.
2. Enter VIXY, SVXY, or cash at the open of day `t+1`.
3. Hold until the open of day `t+2`, when the next EOD signal can be acted upon.

The Backtest page always displays **research close-to-close and next-open-to-next-open side by side**, including CAGR, Sharpe, annualized volatility, maximum drawdown, and both equity curves.

A **next-open → same-close** mode remains available as an intraday diagnostic. Because it exits at each close, transaction costs are charged as a round trip on each invested day.

## Data

- VIX1D, VIX and VIX3M: official Cboe daily-history CSVs.
- VIXY and SVXY: adjusted daily OHLC from Yahoo Finance via `yfinance`.
- VIX1D history begins on 2022-05-13, so that is VixSnap's default research start date.

## Setup

```bash
uv sync
```

Run the dashboard:

```bash
uv run streamlit run app.py
```

Run the tests:

```bash
uv run --with pytest pytest
```

## Dashboard

### Overview

Shows the latest official EOD signal, VIX1D/VIX/VIX3M state, recent signal changes, one-year VIX1D chart, signal mix, and a side-by-side research-vs-executable timing snapshot.

### Backtest

Shows:

- close-to-close research results beside next-open executable results;
- a direct two-equity-curve timing comparison;
- a selectable detailed timing view;
- CAGR, Sharpe, annualized volatility and maximum drawdown;
- exposure and leg diagnostics;
- yearly results;
- downloadable daily backtest output.

### Research

Includes:

- low-VIX1D curve-filter diagnostics;
- a fixed threshold-stability surface for low thresholds 8–12 and high thresholds 18–30;
- selectable timing for the threshold surface, with close-to-close as the default;
- VIX1D versus the `VIX3M - VIX` curve spread.

The threshold surface is intended as a robustness check, not as an optimizer.

## Current research interpretation

The original VIXY/SVXY study suggests two different forms of mean reversion:

- very low VIX1D may identify unusually compressed short-horizon implied volatility that subsequently reprices upward;
- very high VIX1D with `VIX <= VIX3M` may identify short-lived volatility shocks that subsequently mean-revert without the broader curve confirming persistent stress.

VixSnap is a research dashboard, not investment advice.
