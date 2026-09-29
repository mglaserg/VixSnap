#!/usr/bin/env python3
"""
Backtest: short UVXY + short SVXY in a VX-neutral ratio.

CURRENT PRODUCT REGIME
----------------------
Since 2018-02-28:
    UVXY daily target ~= +1.5x S&P 500 VIX Short-Term Futures Index
    SVXY daily target ~= -0.5x S&P 500 VIX Short-Term Futures Index

If we short both:
    short $1 UVXY -> approximately -1.5 units of VX exposure
    short $3 SVXY -> approximately +1.5 units of VX exposure

So the first-order neutral ratio is:
    UVXY : SVXY = 1 : 3 short notional

Normalized to 1.0x gross short exposure:
    UVXY weight = -0.25
    SVXY weight = -0.75

The strategy is rebalanced weekly by default.

IMPORTANT
---------
- This is a market-price backtest, not an exact margin-account simulation.
- Borrow fees are critical and must be stress-tested.
- IBKR margin/buying-power requirements are NOT modeled.
- Adjusted Yahoo prices are used so reverse splits do not create fake returns.
- VIXY is downloaded only as an approximate 1x front-VX benchmark for
  estimating residual directional beta. It is not traded by the pair.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    summary: dict


def download_adjusted_close(symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
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
        raise RuntimeError("Yahoo Finance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            px = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            px = raw["Close"].copy()
        else:
            raise RuntimeError(f"Could not find adjusted/close prices in columns: {raw.columns}")
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        px = raw[[col]].copy()
        if len(symbols) == 1:
            px.columns = symbols

    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px.reindex(columns=symbols)
    return px.dropna(how="any").sort_index()


def is_rebalance_date(index: pd.DatetimeIndex, i: int, freq: str) -> bool:
    if i == 0:
        return True
    if freq == "daily":
        return True
    if freq == "weekly":
        if i == len(index) - 1:
            return True
        return index[i].isocalendar().week != index[i + 1].isocalendar().week
    if freq == "monthly":
        if i == len(index) - 1:
            return True
        return index[i].to_period("M") != index[i + 1].to_period("M")
    raise ValueError(f"Unknown rebalance frequency: {freq}")


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())


def perf_metrics(returns: pd.Series, nav: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {}

    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = np.nan if r.std(ddof=1) == 0 else r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    years = len(r) / TRADING_DAYS
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0 else np.nan

    return {
        "Total Return": nav.iloc[-1] - 1.0,
        "CAGR": cagr,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown(nav),
        "Best Day": r.max(),
        "Worst Day": r.min(),
    }


def target_weights(ratio: float, gross: float) -> dict[str, float]:
    uvxy_abs = gross / (1.0 + ratio)
    svxy_abs = gross * ratio / (1.0 + ratio)
    return {"UVXY": -uvxy_abs, "SVXY": -svxy_abs}


def backtest_pair(
    prices: pd.DataFrame,
    ratio: float = 3.0,
    gross: float = 1.0,
    rebalance: str = "weekly",
    uvxy_borrow: float = 0.0,
    svxy_borrow: float = 0.0,
    cost_bps: float = 5.0,
) -> BacktestResult:
    px = prices[["UVXY", "SVXY", "VIXY"]].copy()
    idx = px.index
    if len(idx) < 3:
        raise ValueError("Not enough observations.")

    weights = target_weights(ratio, gross)
    borrow_rates = {"UVXY": uvxy_borrow, "SVXY": svxy_borrow}
    cost_rate = cost_bps / 10_000.0

    cash = 1.0
    shares = {"UVXY": 0.0, "SVXY": 0.0}
    prev_equity = 1.0

    rows = []
    trades = []

    for i, date in enumerate(idx):
        current_prices = {sym: float(px.loc[date, sym]) for sym in ["UVXY", "SVXY"]}

        borrow_cost = 0.0
        if i > 0:
            prev_date = idx[i - 1]
            for sym in ["UVXY", "SVXY"]:
                prev_mv = shares[sym] * float(px.loc[prev_date, sym])
                if prev_mv < 0:
                    borrow_cost += abs(prev_mv) * borrow_rates[sym] / TRADING_DAYS
            cash -= borrow_cost

        equity_before_rebalance = cash + sum(
            shares[sym] * current_prices[sym] for sym in ["UVXY", "SVXY"]
        )

        if equity_before_rebalance <= 0:
            raise RuntimeError(
                f"Strategy equity became non-positive on {date.date()}. "
                "Lower --gross or inspect the path."
            )

        turnover_notional = 0.0
        txn_cost = 0.0
        rebalanced = is_rebalance_date(idx, i, rebalance)

        if rebalanced:
            for sym in ["UVXY", "SVXY"]:
                target_dollars = weights[sym] * equity_before_rebalance
                target_shares = target_dollars / current_prices[sym]
                delta_shares = target_shares - shares[sym]
                trade_notional = abs(delta_shares) * current_prices[sym]
                txn_cost_sym = trade_notional * cost_rate

                turnover_notional += trade_notional
                txn_cost += txn_cost_sym

                cash -= delta_shares * current_prices[sym]
                cash -= txn_cost_sym

                if abs(delta_shares) > 1e-12:
                    trades.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "price": current_prices[sym],
                            "delta_shares_adjusted_units": delta_shares,
                            "notional": trade_notional,
                            "transaction_cost": txn_cost_sym,
                        }
                    )
                shares[sym] = target_shares

        equity = cash + sum(
            shares[sym] * current_prices[sym] for sym in ["UVXY", "SVXY"]
        )

        daily_return = equity / prev_equity - 1.0 if i > 0 else equity - 1.0

        uvxy_mv = shares["UVXY"] * current_prices["UVXY"]
        svxy_mv = shares["SVXY"] * current_prices["SVXY"]

        rows.append(
            {
                "date": date,
                "UVXY": px.loc[date, "UVXY"],
                "SVXY": px.loc[date, "SVXY"],
                "VIXY": px.loc[date, "VIXY"],
                "equity": equity,
                "return": daily_return,
                "cash": cash,
                "uvxy_market_value": uvxy_mv,
                "svxy_market_value": svxy_mv,
                "gross_short": abs(min(uvxy_mv, 0.0)) + abs(min(svxy_mv, 0.0)),
                "net_market_value": uvxy_mv + svxy_mv,
                "borrow_cost": borrow_cost,
                "transaction_cost": txn_cost,
                "turnover_notional": turnover_notional,
                "rebalanced": rebalanced,
            }
        )
        prev_equity = equity

    daily = pd.DataFrame(rows).set_index("date")
    trade_df = pd.DataFrame(trades)

    vixy_ret = px["VIXY"].pct_change().reindex(daily.index)
    aligned = pd.concat(
        [daily["return"].rename("pair"), vixy_ret.rename("vixy")], axis=1
    ).dropna()

    if len(aligned) > 2 and aligned["vixy"].var() > 0:
        beta_to_vixy = aligned["pair"].cov(aligned["vixy"]) / aligned["vixy"].var()
        corr_to_vixy = aligned["pair"].corr(aligned["vixy"])
    else:
        beta_to_vixy = np.nan
        corr_to_vixy = np.nan

    summary = perf_metrics(daily["return"], daily["equity"])
    summary.update(
        {
            "Ratio SVXY/UVXY": ratio,
            "Gross Target": gross,
            "Rebalance": rebalance,
            "UVXY Borrow Ann.": uvxy_borrow,
            "SVXY Borrow Ann.": svxy_borrow,
            "Trading Cost bps": cost_bps,
            "Beta to VIXY": beta_to_vixy,
            "Corr to VIXY": corr_to_vixy,
            "Total Borrow Cost / Initial Equity": daily["borrow_cost"].sum(),
            "Total Trading Cost / Initial Equity": daily["transaction_cost"].sum(),
            "Total Turnover / Initial Equity": daily["turnover_notional"].sum(),
            "Rebalances": int(daily["rebalanced"].sum()),
        }
    )

    return BacktestResult(daily=daily, trades=trade_df, summary=summary)


def simple_benchmark_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.pct_change()
    out = pd.DataFrame(index=prices.index)
    out["Long SVXY"] = r["SVXY"]
    out["Short UVXY 1x"] = -r["UVXY"]
    out["Long VIXY"] = r["VIXY"]
    return out


def print_summary(summary: dict) -> None:
    pct_fields = {
        "Total Return", "CAGR", "Ann Vol", "Max Drawdown",
        "Best Day", "Worst Day", "UVXY Borrow Ann.", "SVXY Borrow Ann.",
        "Total Borrow Cost / Initial Equity",
        "Total Trading Cost / Initial Equity",
    }

    print("\n=== SHORT UVXY + SHORT SVXY ===")
    for k, v in summary.items():
        if isinstance(v, (float, np.floating)):
            if k in pct_fields:
                print(f"{k:38s} {v:>11.2%}")
            elif k in {"Sharpe", "Beta to VIXY", "Corr to VIXY"}:
                print(f"{k:38s} {v:>11.3f}")
            else:
                print(f"{k:38s} {v:>11.4f}")
        else:
            print(f"{k:38s} {v}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2018-02-28")
    p.add_argument("--end", default=None)
    p.add_argument("--ratio", type=float, default=3.0)
    p.add_argument("--gross", type=float, default=1.0)
    p.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="weekly")
    p.add_argument("--uvxy-borrow", type=float, default=0.0)
    p.add_argument("--svxy-borrow", type=float, default=0.0)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--outdir", default="output/neutral_vol_etp")
    args = p.parse_args()

    prices = download_adjusted_close(["UVXY", "SVXY", "VIXY"], args.start, args.end)

    result = backtest_pair(
        prices=prices,
        ratio=args.ratio,
        gross=args.gross,
        rebalance=args.rebalance,
        uvxy_borrow=args.uvxy_borrow,
        svxy_borrow=args.svxy_borrow,
        cost_bps=args.cost_bps,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result.daily.to_csv(outdir / "daily.csv")
    result.trades.to_csv(outdir / "trades.csv", index=False)
    pd.Series(result.summary, name="value").to_csv(outdir / "summary.csv")

    benchmarks = simple_benchmark_returns(prices).reindex(result.daily.index)
    benchmark_rows = []
    for name in benchmarks.columns:
        rr = benchmarks[name].fillna(0.0)
        nav = (1.0 + rr).cumprod()
        m = perf_metrics(rr, nav)
        m["Strategy"] = name
        benchmark_rows.append(m)

    bench_df = pd.DataFrame(benchmark_rows).set_index("Strategy")
    bench_df.to_csv(outdir / "benchmarks.csv")

    sensitivity_rows = []
    for borrow_rate in [0.00, 0.05, 0.10, 0.20, 0.30]:
        s = backtest_pair(
            prices=prices,
            ratio=args.ratio,
            gross=args.gross,
            rebalance=args.rebalance,
            uvxy_borrow=borrow_rate,
            svxy_borrow=borrow_rate,
            cost_bps=args.cost_bps,
        )
        sensitivity_rows.append(
            {
                "Borrow rate each leg": borrow_rate,
                "CAGR": s.summary.get("CAGR"),
                "Sharpe": s.summary.get("Sharpe"),
                "Max Drawdown": s.summary.get("Max Drawdown"),
                "Total Return": s.summary.get("Total Return"),
            }
        )

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(outdir / "borrow_sensitivity.csv", index=False)

    ratio_rows = []
    for ratio in np.arange(2.0, 4.01, 0.25):
        s = backtest_pair(
            prices=prices,
            ratio=float(ratio),
            gross=args.gross,
            rebalance=args.rebalance,
            uvxy_borrow=args.uvxy_borrow,
            svxy_borrow=args.svxy_borrow,
            cost_bps=args.cost_bps,
        )
        ratio_rows.append(
            {
                "SVXY/UVXY ratio": ratio,
                "CAGR": s.summary.get("CAGR"),
                "Sharpe": s.summary.get("Sharpe"),
                "Max Drawdown": s.summary.get("Max Drawdown"),
                "Beta to VIXY": s.summary.get("Beta to VIXY"),
                "Corr to VIXY": s.summary.get("Corr to VIXY"),
            }
        )

    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(outdir / "ratio_diagnostic.csv", index=False)

    print_summary(result.summary)

    print("\n=== BENCHMARKS ===")
    pretty_bench = bench_df.copy()
    for c in ["Total Return", "CAGR", "Ann Vol", "Max Drawdown", "Best Day", "Worst Day"]:
        if c in pretty_bench:
            pretty_bench[c] = pretty_bench[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")
    if "Sharpe" in pretty_bench:
        pretty_bench["Sharpe"] = pretty_bench["Sharpe"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
    print(pretty_bench.to_string())

    print("\n=== BORROW SENSITIVITY ===")
    pretty_sens = sens_df.copy()
    for c in ["Borrow rate each leg", "CAGR", "Max Drawdown", "Total Return"]:
        pretty_sens[c] = pretty_sens[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")
    pretty_sens["Sharpe"] = pretty_sens["Sharpe"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
    print(pretty_sens.to_string(index=False))

    print("\n=== RATIO / VX-NEUTRALITY DIAGNOSTIC ===")
    pretty_ratio = ratio_df.copy()
    for c in ["CAGR", "Max Drawdown"]:
        pretty_ratio[c] = pretty_ratio[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")
    for c in ["Sharpe", "Beta to VIXY", "Corr to VIXY"]:
        pretty_ratio[c] = pretty_ratio[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
    print(pretty_ratio.to_string(index=False))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(11, 6))
        plt.plot(result.daily.index, result.daily["equity"], label="Short UVXY + Short SVXY")
        for name in ["Long SVXY", "Short UVXY 1x"]:
            nav = (1.0 + benchmarks[name].fillna(0.0)).cumprod()
            plt.plot(nav.index, nav, label=name, alpha=0.65)
        plt.yscale("log")
        plt.title(
            f"Neutral Vol ETP Pair — ratio {args.ratio:g}:1, "
            f"{args.rebalance} rebalance, gross {args.gross:g}x"
        )
        plt.ylabel("Growth of $1 (log scale)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "equity.png", dpi=160)
        plt.close()

        dd = result.daily["equity"] / result.daily["equity"].cummax() - 1.0
        plt.figure(figsize=(11, 5))
        plt.plot(dd.index, dd)
        plt.title("Pair Strategy Drawdown")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(outdir / "drawdown.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"\nPlotting skipped: {exc}")

    print(
        "\nNOTE: Borrow sensitivity uses the SAME annual borrow rate on both legs "
        "only as a stress test. For a serious result, use historical or broker-"
        "specific UVXY and SVXY borrow rates separately."
    )
    print(
        "IBKR margin requirements / buying power are not modeled. A strategy "
        "can look attractive per dollar of equity while being unattractive per "
        "dollar of required buying power."
    )
    print(f"\nSaved results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
