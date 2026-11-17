#!/usr/bin/env python3
"""
Signal Lab (Python) - real-data indicators + honest backtest.
"""
import argparse, math, sys, time
import numpy as np
import pandas as pd

PAYOUT = 0.92
BREAK_EVEN = 1 / (1 + PAYOUT)

def load_data(symbol, interval, period, demo=False):
    if demo:
        rng = np.random.default_rng(7)
        steps = rng.normal(0, 0.0004, 6000)
        close = 1.09 + np.cumsum(steps)
        idx = pd.date_range("2024-01-01", periods=len(close), freq="5min")
        return pd.DataFrame({"Close": close}, index=idx)
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Missing package: pip install yfinance pandas numpy")
    df = yf.download(symbol, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if df.empty:
        sys.exit("No data returned. Check the symbol / interval / period.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).sum()
    loss = (-d.clip(upper=0)).rolling(n).sum()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def add_signals(df):
    c = df["Close"]
    ema = lambda n: c.ewm(span=n, adjust=False).mean()
    df = df.copy()
    df["v_ema_fast"] = np.where(ema(20) > ema(50), 1, -1)
    df["v_ema_slow"] = np.where(ema(50) > ema(200), 1, -1)
    df["v_mom"] = np.where(c - c.shift(5) > 0, 1, -1)
    df["rsi"] = rsi(c)
    df["v_rsi"] = np.where(df["rsi"] > 50, 1, -1)
    df["score"] = df[["v_ema_fast", "v_ema_slow", "v_mom", "v_rsi"]].sum(axis=1)
    df["signal"] = np.sign(df["score"]).astype(int)
    return df.iloc[200:]

def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half

def p_value_vs_breakeven(w, n):
    if n == 0:
        return 1.0
    p0 = BREAK_EVEN
    z = (w / n - p0) / math.sqrt(p0 * (1 - p0) / n)
    return 0.5 * math.erfc(z / math.sqrt(2))

def backtest(df, stake=10.0):
    nxt = df["Close"].shift(-1)
    d = df.assign(next=nxt).dropna()
    d = d[(d["signal"] != 0) & (d["next"] != d["Close"])]
    win = ((d["signal"] > 0) & (d["next"] > d["Close"])) | \
          ((d["signal"] < 0) & (d["next"] < d["Close"]))
    pnl = np.where(win, stake * PAYOUT, -stake)
    return win.to_numpy(), pnl, d.index

def report(name, win, pnl):
    n, w = len(win), int(win.sum())
    lo, hi = wilson(w, n)
    eq = np.cumsum(pnl)
    dd = (np.maximum.accumulate(eq) - eq).max() if n else 0
    print(f"\n[{name}]")
    print(f"  signals        : {n}")
    print(f"  wins           : {w}  ({100*w/n:.2f}%)" if n else "  no signals")
    print(f"  95% CI         : {100*lo:.1f}% - {100*hi:.1f}%")
    print(f"  break-even     : {100*BREAK_EVEN:.2f}%  (payout {PAYOUT:.0%})")
    print(f"  p-value (>BE)  : {p_value_vs_breakeven(w, n):.3f}")
    print(f"  net P/L        : {eq[-1] if n else 0:+.2f}   max drawdown: {dd:.2f}")
    if n:
        verdict = ("CI includes break-even -> no proven edge"
                   if lo <= BREAK_EVEN else "above break-even on this sample")
        print(f"  verdict        : {verdict}")
    return eq

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="EURUSD=X")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--period", default="30d")
    ap.add_argument("--demo", action="store_true", help="use random data")
    ap.add_argument("--live", action="store_true", help="print latest signal repeatedly")
    ap.add_argument("--plot", action="store_true", help="save equity curve PNG")
    a = ap.parse_args()

    raw = load_data(a.symbol, a.interval, a.period, a.demo)
    df = add_signals(raw)
    print(f"Data: {a.symbol if not a.demo else 'DEMO (random)'}  "
          f"{a.interval}  {len(raw)} candles  {raw.index[0]} -> {raw.index[-1]}")

    win, pnl, idx = backtest(df)
    half = len(win) // 2
    eq_all = report("Full sample", win, pnl)
    report("First half (in-sample)", win[:half], pnl[:half])
    report("Second half (out-of-sample)", win[half:], pnl[half:])

    if a.plot and len(pnl):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(idx, eq_all)
        plt.axhline(0, color="gray", lw=.8)
        plt.title("Paper equity (stake 10, payout 92%)")
        plt.tight_layout()
        plt.savefig("equity.png", dpi=130)
        print("Saved equity.png")

    last = df.iloc[-1]
    names = {1: "BUY", -1: "SELL", 0: "WAIT"}
    print(f"\nLatest candle {df.index[-1]}: {names[int(last['signal'])]}  "
          f"(score {int(last['score']):+d}, RSI {last['rsi']:.1f})")

    if a.live:
        print("Live mode - Ctrl+C to stop. Analysis only, no trading.")
        try:
            while True:
                time.sleep(60)
                df = add_signals(load_data(a.symbol, a.interval, a.period))
                r = df.iloc[-1]
                print(f"{df.index[-1]}  {names[int(r['signal'])]}  "
                      f"score {int(r['score']):+d}  RSI {r['rsi']:.1f}")
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
