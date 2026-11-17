#!/usr/bin/env python3
"""
Signal Lab Local API Server - Pro Ultra Engine V3 + Ollama Bridge
Includes:
- Engine V2/V3 Analytics (ADX, Bollinger Bands, Candlestick Patterns, MTF Confirmation)
- Parallel Top Signals Scanner (/api/scan) across all 56 OTC assets
- Automated Paper Trade Settlement (auto-calculates profit/loss after expiry)
- Local Ollama AI Bridge (/api/ollama)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing package: pip install yfinance pandas numpy flask flask-cors requests")
app = Flask(__name__)
CORS(app)

DB_NAME = "signal_lab.db"
CACHE = {}
CACHE_TTL = 8  # seconds
INITIAL_BALANCE = 1000.0
PAYOUT = 0.92

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    signal TEXT,
                    price REAL,
                    close_price REAL,
                    risk_amount REAL,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'OPEN',
                    duration_sec INTEGER DEFAULT 60,
                    timestamp TEXT
                )''')
    c.execute("PRAGMA table_info(trades)")
    cols = [row[1] for row in c.fetchall()]
    if "close_price" not in cols:
        c.execute("ALTER TABLE trades ADD COLUMN close_price REAL")
    if "pnl" not in cols:
        c.execute("ALTER TABLE trades ADD COLUMN pnl REAL DEFAULT 0")
    if "status" not in cols:
        c.execute("ALTER TABLE trades ADD COLUMN status TEXT DEFAULT 'OPEN'")
    if "duration_sec" not in cols:
        c.execute("ALTER TABLE trades ADD COLUMN duration_sec INTEGER DEFAULT 60")
    conn.commit()
    conn.close()

init_db()

ASSETS_DATA = {
    "AED/CNY": ("AEDCNY_otc", "🇦🇪🇨🇳"), "AUD/USD": ("AUDUSD_otc", "🇦🇺🇺🇸"),
    "CAD/JPY": ("CADJPY_otc", "🇨🇦🇯🇵"), "EUR/HUF": ("EURHUF_otc", "🇪🇺🇭🇺"),
    "EUR/JPY": ("EURJPY_otc", "🇪🇺🇯🇵"), "EUR/USD": ("EURUSD_otc", "🇪🇺🇺🇸"),
    "GBP/JPY": ("GBPJPY_otc", "🇬🇧🇯🇵"), "GBP/USD": ("GBPUSD_otc", "🇬🇧🇺🇸"),
    "NZD/USD": ("NZDUSD_otc", "🇳🇿🇺🇸"), "SAR/CNY": ("SARCNY_otc", "🇸🇦🇨🇳"),
    "USD/CHF": ("USDCHF_otc", "🇺🇸🇨🇭"), "USD/CNH": ("USDCNH_otc", "🇺🇸🇨🇳"),
    "USD/IDR": ("USDIDR_otc", "🇺🇸🇮🇩"), "USD/PHP": ("USDPHP_otc", "🇺🇸🇵🇭"),
    "USD/VND": ("USDVND_otc", "🇺🇸🇻🇳"), "YER/USD": ("YERUSD_otc", "🇾🇪🇺🇸"),
    "CAD/CHF": ("CADCHF_otc", "🇨🇦🇨🇭"), "AUD/JPY": ("AUDJPY_otc", "🇦🇺🇯🇵"),
    "EUR/RUB": ("EURRUB_otc", "🇪🇺🇷🇺"), "QAR/CNY": ("QARCNY_otc", "🇶🇦🇨🇳"),
    "USD/JPY": ("USDJPY_otc", "🇺🇸🇯🇵"), "BHD/CNY": ("BHDCNY_otc", "🇧🇭🇨🇳"),
    "AUD/CHF": ("AUDCHF_otc", "🇦🇺🇨🇭"), "EUR/CHF": ("EURCHF_otc", "🇪🇺🇨🇭"),
    "NGN/USD": ("NGNUSD_otc", "🇳🇬🇺🇸"), "USD/BDT": ("USDBDT_otc", "🇺🇸🇧🇩"),
    "USD/PKR": ("USDPKR_otc", "🇺🇸🇵🇰"), "USD/CLP": ("USDCLP_otc", "🇺🇸🇨🇱"),
    "CHF/NOK": ("CHFNOK_otc", "🇨🇭🇳🇴"), "TND/USD": ("TNDUSD_otc", "🇹🇳🇺🇸"),
    "MAD/USD": ("MADUSD_otc", "🇲🇦🇺🇸"), "USD/CAD": ("USDCAD_otc", "🇺🇸🇨🇦"),
    "USD/EGP": ("USDEGP_otc", "🇺🇸🇪🇬"), "AUD/CAD": ("AUDCAD_otc", "🇦🇺🇨🇦"),
    "ZAR/USD": ("ZARUSD_otc", "🇿🇦🇺🇸"), "USD/ARS": ("USDARS_otc", "🇺🇸🇦🇷"),
    "EUR/NZD": ("EURNZD_otc", "🇪🇺🇳🇿"), "USD/MXN": ("USDMXN_otc", "🇺🇸🇲🇽"),
    "KES/USD": ("KESUSD_otc", "🇰🇪🇺🇸"), "USD/COP": ("USDCOP_otc", "🇺🇸🇨🇴"),
    "USD/BRL": ("USDBRL_otc", "🇺🇸🇧🇷"), "EUR/TRY": ("EURTRY_otc", "🇪🇺🇹🇷"),
    "USD/SGD": ("USDSGD_otc", "🇺🇸🇸🇬"), "USD/MYR": ("USDMYR_otc", "🇺🇸🇲🇾"),
    "CHF/JPY": ("CHFJPY_otc", "🇨🇭🇯🇵"), "USD/RUB": ("USDRUB_otc", "🇺🇸🇷🇺"),
    "AUD/NZD": ("AUDNZD_otc", "🇦🇺🇳🇿"), "NZD/JPY": ("NZDJPY_otc", "🇳🇿🇯🇵"),
    "USD/DZD": ("USDDZD_otc", "🇺🇸🇩🇿"), "JOD/CNY": ("JODCNY_otc", "🇯🇴🇨🇳"),
    "LBP/USD": ("LBPUSD_otc", "🇱🇧🇺🇸"), "USD/INR": ("USDINR_otc", "🇺🇸🇮🇳"),
    "GBP/AUD": ("GBPAUD_otc", "🇬🇧🇦🇺"), "OMR/CNY": ("OMRCNY_otc", "🇴🇲🇨🇳"),
    "JOD/THB": ("JODTHB_otc", "🇯🇴🇹🇭"), "UAH/USD": ("UAHUSD_otc", "🇺🇦🇺🇸"),
}

TF_MAP = {
    "S3":  ("1m", "2d", "1m (بديل S3)", "5m", "15d", 3),
    "S15": ("1m", "2d", "1m (بديل S15)", "5m", "15d", 15),
    "S30": ("1m", "2d", "1m (بديل S30)", "5m", "15d", 30),
    "M1":  ("1m", "5d", "1m", "5m", "15d", 60),
    "M3":  ("1m", "5d", "3m", "15m", "30d", 180),
    "M5":  ("5m", "30d", "5m", "15m", "60d", 300),
    "M10": ("5m", "30d", "10m", "30m", "60d", 600),
    "M15": ("15m", "60d", "15m", "1h", "90d", 900),
    "M30": ("15m", "60d", "30m", "1h", "90d", 1800),
}

def get_yf_symbol(otc_code):
    base = otc_code.replace("_otc", "")
    mapping = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
        "NZDUSD": "NZDUSD=X", "GBPJPY": "GBPJPY=X", "EURJPY": "EURJPY=X",
        "CADJPY": "CADJPY=X", "AUDJPY": "AUDJPY=X", "CHFJPY": "CHFJPY=X",
        "EURCHF": "EURCHF=X", "AUDCAD": "AUDCAD=X", "EURNZD": "EURNZD=X",
        "GBPAUD": "GBPAUD=X", "EURTRY": "EURTRY=X", "USDSGD": "USDSGD=X",
        "USDMXN": "USDMXN=X", "USDZAR": "ZARUSD=X", "USDBRL": "USDBRL=X",
        "USDINR": "USDINR=X", "USDRUB": "RUB=X", "USDUAH": "UAH=X",
        "AUDNZD": "AUDNZD=X", "NZDJPY": "NZDJPY=X", "CADCHF": "CADCHF=X",
        "AUDCHF": "AUDCHF=X", "CHFNOK": "CHFNOK=X"
    }
    if base in mapping:
        return mapping[base]
    if len(base) == 6:
        return base[:3] + base[3:] + "=X"
    return "EURUSD=X"

def load_data(otc_symbol="EURUSD_otc", interval="5m", period="30d"):
    cache_key = f"{otc_symbol}_{interval}_{period}"
    now = time.time()
    if cache_key in CACHE:
        ts, cached_df, is_live = CACHE[cache_key]
        if now - ts < CACHE_TTL:
            return cached_df.copy(), is_live

    yf_sym = get_yf_symbol(otc_symbol)
    try:
        df = yf.download(yf_sym, interval=interval, period=period,
                         auto_adjust=True, progress=False)
        if not df.empty and "Close" in df.columns:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            res = df[["Open", "High", "Low", "Close"]].dropna()
            if len(res) > 210:
                CACHE[cache_key] = (now, res, True)
                return res.copy(), True
    except Exception:
        pass
    
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}
    pd_freq = freq_map.get(interval, "5min")
    
    seed = (hash(otc_symbol) + int(now // 10)) % 2**32
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0004, 600)
    close = 1.1000 + np.cumsum(steps)
    high = close + abs(rng.normal(0, 0.00025, 600))
    low = close - abs(rng.normal(0, 0.00025, 600))
    open_p = close + rng.normal(0, 0.00015, 600)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=len(close), freq=pd_freq)
    sim_df = pd.DataFrame({"Open": open_p, "High": high, "Low": low, "Close": close}, index=idx)
    CACHE[cache_key] = (now, sim_df, False)
    return sim_df.copy(), False

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def compute_adx(high, low, close, n=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (pd.Series(plus_dm, index=close.index).ewm(alpha=1/n, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=close.index).ewm(alpha=1/n, adjust=False).mean() / atr)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx.fillna(20.0), plus_di.fillna(0.0), minus_di.fillna(0.0)

def detect_candle_pattern(df):
    if len(df) < 3:
        return 0, "NONE"
    c1 = df.iloc[-1]
    c0 = df.iloc[-2]

    body1 = abs(c1["Close"] - c1["Open"])
    rng1 = max(c1["High"] - c1["Low"], 1e-9)
    upper_wick = c1["High"] - max(c1["Open"], c1["Close"])
    lower_wick = min(c1["Open"], c1["Close"]) - c1["Low"]
    body0 = abs(c0["Close"] - c0["Open"])

    if lower_wick > 2.2 * body1 and upper_wick < 0.3 * rng1:
        return 1, "PIN_BULL"
    if upper_wick > 2.2 * body1 and lower_wick < 0.3 * rng1:
        return -1, "PIN_BEAR"

    if c0["Close"] < c0["Open"] and c1["Close"] > c1["Open"] and body1 > body0 * 1.15:
        if c1["Close"] >= c0["Open"] and c1["Open"] <= c0["Close"]:
            return 1, "ENG_BULL"

    if c0["Close"] > c0["Open"] and c1["Close"] < c1["Open"] and body1 > body0 * 1.15:
        if c1["Close"] <= c0["Open"] and c1["Open"] >= c0["Close"]:
            return -1, "ENG_BEAR"

    return 0, "NEUTRAL"

def get_mtf_bias(otc_symbol, htf_interval, htf_period):
    try:
        htf_df, _ = load_data(otc_symbol, htf_interval, htf_period)
        c = htf_df["Close"]
        ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        rsi_htf = rsi(c, 14).iloc[-1]
        if ema20 > ema50 and rsi_htf > 51:
            return 1
        elif ema20 < ema50 and rsi_htf < 49:
            return -1
    except Exception:
        pass
    return 0

def add_signals(df, mtf_bias=0):
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    ema = lambda n: c.ewm(span=n, adjust=False).mean()
    df = df.copy()
    
    df["ema20"] = ema(20)
    df["ema50"] = ema(50)
    df["ema200"] = ema(200)
    
    ema12 = ema(12)
    ema26 = ema(26)
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    raw_k = 100 * ((c - low14) / (high14 - low14).replace(0, np.nan))
    df["stoch_k"] = raw_k.rolling(3).mean()
    
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["adx"], df["plus_di"], df["minus_di"] = compute_adx(h, l, c, 14)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    df["rsi"] = rsi(c)

    df["v_ema_fast"] = np.where(df["ema20"] > df["ema50"], 1, -1)
    df["v_ema_slow"] = np.where(df["ema50"] > df["ema200"], 1, -1)
    df["v_mom"] = np.where(c - c.shift(5) > 0, 1, -1)
    df["v_rsi"] = np.where(df["rsi"] > 50, 1, -1)
    df["v_macd"] = np.where(df["macd"] > df["macd_signal"], 1, -1)
    df["v_stoch"] = np.where(df["stoch_k"] > 50, 1, -1)
    df["v_mtf"] = mtf_bias

    df["score"] = (
        df["v_ema_fast"] * 1.5 +
        df["v_ema_slow"] * 2.0 +
        df["v_mom"] * 1.0 +
        df["v_rsi"] * 1.0 +
        df["v_macd"] * 1.5 +
        df["v_stoch"] * 1.0 +
        df["v_mtf"] * 2.0
    )
    df["signal"] = np.sign(df["score"]).astype(int)
    return df.iloc[200:]

def evaluate_symbol(symbol, tf_key="M1"):
    interval, period, label_note, htf_int, htf_per, _ = TF_MAP.get(tf_key, ("1m", "5d", "1m", "5m", "15d", 60))
    raw, is_live = load_data(otc_symbol=symbol, interval=interval, period=period)
    mtf_bias = get_mtf_bias(symbol, htf_int, htf_per)
    df = add_signals(raw, mtf_bias=mtf_bias)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    pat_vote, pat_code = detect_candle_pattern(raw.tail(5))
    
    recent_closes = [0.0 if math.isnan(x) else float(x) for x in df["Close"].tail(30).tolist()]
    w30 = df.tail(30)
    hi = float(w30["High"].max()) if not w30.empty else float(last["Close"])
    lo = float(w30["Low"].min()) if not w30.empty else float(last["Close"])
    price = float(last["Close"])
    atr_val = float(last["atr"]) if not math.isnan(last["atr"]) else 0.0010
    adx_val = float(last["adx"])
    bb_pos = float(last["bb_pos"])
    rsi_val = float(last["rsi"])
    if math.isnan(rsi_val) or math.isinf(rsi_val):
        rsi_val = 50.0
    
    pip = 0.0001
    d_above = round((hi - price) / pip, 1)
    d_below = round((price - lo) / pip, 1)
    move_pips = round((price - float(prev["Close"])) / pip, 1)
    
    h = datetime.now(timezone.utc).hour
    sess_key = "tok" if h < 7 else ("lon" if h < 12 else ("ny" if h < 21 else "syd"))
    
    raw_score = float(last["score"]) + (pat_vote * 1.5)
    filter_reason = ""

    if adx_val < 18.0:
        raw_score *= 0.25
        filter_reason = "LOW_ADX"

    if raw_score > 0 and (bb_pos > 0.96 or rsi_val > 76):
        raw_score = 0.0
        filter_reason = "OVERBOUGHT_BB"
    elif raw_score < 0 and (bb_pos < 0.04 or rsi_val < 24):
        raw_score = 0.0
        filter_reason = "OVERSOLD_BB"

    if mtf_bias == 1 and raw_score < 0 and abs(raw_score) < 6.0:
        raw_score = 0.0
        filter_reason = "MTF_GUARD"
    elif mtf_bias == -1 and raw_score > 0 and abs(raw_score) < 6.0:
        raw_score = 0.0
        filter_reason = "MTF_GUARD"

    max_possible_score = 11.5
    strength_pct = min(100.0, round((abs(raw_score) / max_possible_score) * 100, 1))
    sig = "BUY" if raw_score >= 2.2 else ("SELL" if raw_score <= -2.2 else "WAIT")

    return {
        "ok": True,
        "symbol": symbol,
        "is_live": is_live,
        "tf_note": label_note,
        "signal": sig,
        "score": round(raw_score, 2),
        "strength_pct": strength_pct,
        "filter_reason": filter_reason,
        "v_ema_fast": int(last["v_ema_fast"]),
        "v_ema_slow": int(last["v_ema_slow"]),
        "v_mom": int(last["v_mom"]),
        "v_rsi": int(last["v_rsi"]),
        "v_stoch": int(last["v_stoch"]),
        "v_mtf": int(mtf_bias),
        "adx": round(adx_val, 1),
        "bb_pos": round(bb_pos * 100, 1),
        "pattern": pat_code,
        "rsi": round(rsi_val, 1),
        "stoch": round(float(last["stoch_k"]), 1),
        "atr": round(atr_val * 10000, 1),
        "move_pips": 0.0 if math.isnan(move_pips) else move_pips,
        "d_above": 0.0 if math.isnan(d_above) else d_above,
        "d_below": 0.0 if math.isnan(d_below) else d_below,
        "session": sess_key,
        "price": price,
        "history": recent_closes,
        "timestamp": str(df.index[-1])
    }

@app.route("/api/assets")
def get_assets():
    formatted = [{"name": k, "code": v[0], "flag": v[1]} for k, v in ASSETS_DATA.items()]
    return jsonify({"ok": True, "assets": formatted})

@app.route("/api/signal")
def get_signal():
    tf_key = request.args.get("tf", "M1")
    symbol = request.args.get("symbol", "EURUSD_otc")
    try:
        res = evaluate_symbol(symbol, tf_key)
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/scan")
def scan_top_opportunities():
    tf_key = request.args.get("tf", "M1")
    results = []
    name_lookup = {v[0]: (k, v[1]) for k, v in ASSETS_DATA.items()}

    def task(code):
        try:
            data = evaluate_symbol(code, tf_key)
            if data["signal"] in ("BUY", "SELL"):
                pair_name, flag = name_lookup.get(code, (code, "💱"))
                return {
                    "symbol": code,
                    "name": pair_name,
                    "flag": flag,
                    "signal": data["signal"],
                    "strength_pct": data["strength_pct"],
                    "price": data["price"],
                    "adx": data["adx"],
                    "is_live": data["is_live"]
                }
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(task, v[0]) for v in ASSETS_DATA.values()]
        for fut in as_completed(futures):
            item = fut.result()
            if item:
                results.append(item)

    results.sort(key=lambda x: (x["strength_pct"], x["is_live"]), reverse=True)
    return jsonify({"ok": True, "top": results[:5], "scanned": len(ASSETS_DATA)})

def settle_open_trades():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, symbol, signal, price, risk_amount, duration_sec, timestamp FROM trades WHERE status = 'OPEN'")
    open_rows = c.fetchall()
    now_utc = datetime.now(timezone.utc)

    for row in open_rows:
        tid, sym, sig, entry_price, risk_amt, dur_sec, ts_str = row
        try:
            opened_at = datetime.fromisoformat(ts_str)
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
        except Exception:
            opened_at = now_utc - timedelta(seconds=120)

        dur = dur_sec or 60
        if (now_utc - opened_at).total_seconds() >= dur:
            try:
                raw, _ = load_data(otc_symbol=sym, interval="1m", period="1d")
                current_price = float(raw["Close"].iloc[-1])
            except Exception:
                current_price = entry_price

            is_win = (sig == "BUY" and current_price > entry_price) or (sig == "SELL" and current_price < entry_price)
            pnl = round(risk_amt * PAYOUT, 2) if is_win else round(-risk_amt, 2)
            status = "WIN" if is_win else "LOSS"

            c.execute("UPDATE trades SET close_price = ?, pnl = ?, status = ? WHERE id = ?",
                      (current_price, pnl, status, tid))

    conn.commit()
    conn.close()

@app.route("/api/trades", methods=["GET", "POST", "DELETE"])
def handle_trades():
    settle_open_trades()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if request.method == "POST":
        data = request.json
        tf_key = data.get("tf", "M1")
        dur_sec = TF_MAP.get(tf_key, ("1m", "5d", "1m", "5m", "15d", 60))[5]
        c.execute("""INSERT INTO trades (symbol, signal, price, risk_amount, status, duration_sec, timestamp)
                     VALUES (?, ?, ?, ?, 'OPEN', ?, ?)""",
                  (data.get("symbol"), data.get("signal"), data.get("price"),
                   data.get("risk_amount"), dur_sec, str(datetime.now(timezone.utc))))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    elif request.method == "DELETE":
        c.execute("DELETE FROM trades")
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    else:
        c.execute("""SELECT symbol, signal, price, close_price, risk_amount, pnl, status, duration_sec, timestamp
                     FROM trades ORDER BY id DESC LIMIT 50""")
        rows = c.fetchall()
        
        c.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status IN ('WIN', 'LOSS')")
        realized_pnl = float(c.fetchone()[0] or 0.0)
        c.execute("SELECT COALESCE(SUM(risk_amount), 0) FROM trades WHERE status = 'OPEN'")
        locked_open = float(c.fetchone()[0] or 0.0)
        c.execute("SELECT COUNT(*) FROM trades WHERE status = 'WIN'")
        wins = int(c.fetchone()[0] or 0)
        c.execute("SELECT COUNT(*) FROM trades WHERE status IN ('WIN', 'LOSS')")
        settled_total = int(c.fetchone()[0] or 0)
        conn.close()

        balance = round(INITIAL_BALANCE + realized_pnl - locked_open, 2)
        win_rate = round((wins / settled_total) * 100, 1) if settled_total > 0 else None

        trades = [{
            "symbol": r[0],
            "signal": r[1],
            "price": r[2],
            "close_price": r[3],
            "risk": r[4],
            "pnl": r[5],
            "status": r[6] or "OPEN",
            "duration": r[7] or 60,
            "time": r[8]
        } for r in rows]

        return jsonify({
            "ok": True,
            "balance": balance,
            "win_rate": win_rate,
            "settled_total": settled_total,
            "wins": wins,
            "trades": trades
        })

@app.route("/api/backtest")
def get_backtest():
    tf_key = request.args.get("tf", "M5")
    symbol = request.args.get("symbol", "EURUSD_otc")
    interval, period, _, htf_int, htf_per, _ = TF_MAP.get(tf_key, ("5m", "30d", "5m", "15m", "60d", 300))
    
    try:
        raw, _ = load_data(otc_symbol=symbol, interval=interval, period=period)
        mtf_bias = get_mtf_bias(symbol, htf_int, htf_per)
        df = add_signals(raw, mtf_bias=mtf_bias)
        
        filtered_sig = np.where(
            (df["adx"] >= 18.0) &
            ~((df["score"] > 0) & ((df["bb_pos"] > 0.96) | (df["rsi"] > 76))) &
            ~((df["score"] < 0) & ((df["bb_pos"] < 0.04) | (df["rsi"] < 24))) &
            (df["score"].abs() >= 2.2),
            np.sign(df["score"]),
            0
        )
        df["filtered_sig"] = filtered_sig

        nxt = df["Close"].shift(-1)
        d = df.assign(next=nxt).dropna()
        d = d[(d["filtered_sig"] != 0) & (d["next"] != df["Close"])]
        win = ((d["filtered_sig"] > 0) & (d["next"] > d["Close"])) | ((d["filtered_sig"] < 0) & (d["next"] < d["Close"]))
        n = int(len(win))
        w = int(win.sum())
        wr = round(w / n * 100, 1) if n else 0.0
        return jsonify({
            "ok": True,
            "signals": n,
            "wins": w,
            "win_rate": wr
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/ollama", methods=["POST"])
def ollama_bridge():
    """جسر ربط مع نموذج Ollama المحلي لتحليل السوق وزيادة دقة القرارات"""
    data = request.json or {}
    prompt = data.get("prompt", "Analyze this trading market data.")
    model = data.get("model", "llama3")
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        
        if response.status_code == 200:
            res_json = response.json()
            return jsonify({"ok": True, "analysis": res_json.get("response", "No response")})
        else:
            return jsonify({"ok": False, "error": f"Ollama status code {response.status_code}"})
    except Exception as e:
        return jsonify({"ok": False, "error": "تعذر الاتصال بـ Ollama المحلي. تأكد من تشغيله عبر الأمر ollama serve."})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
