# ============================================================
# FILE: crypto_alert_bot.py
# Lean market-structure scanner with Telegram alerts.
#
# WHAT CHANGED FROM THE ORIGINAL v21 SCRIPT:
#   - No Excel workbook, no local trade journal / outcome tracking
#   - No CoinGecko dependency — universe is built from the
#     exchange's own 24h volume ranking + a static watchlist
#   - Sends a Telegram message for new A+/A tier signals
#   - Keeps a small JSON "state" file to avoid re-alerting the
#     same setup every run (designed to be committed back to the
#     repo by the GitHub Actions workflow)
#
# DATA SOURCE: Kraken's public REST API (spot USDT pairs). Both Binance
# and Bybit's futures APIs return HTTP 451/403 to requests coming from
# major cloud-provider IP ranges (AWS/GCP/Azure) — which is exactly what
# GitHub Actions runners use — so neither can be called reliably from a
# scheduled GitHub workflow. Kraken is a US-licensed, heavily regulated
# exchange built to be broadly accessible, including to cloud/CI traffic,
# and its public market-data endpoints require no API key. Note this uses
# SPOT prices, not perpetual futures — funding rate / open interest fields
# from the original design aren't available here, but structure, trend,
# sweep, OB, and FVG detection all work identically on spot candles.
#
# IMPORTANT: This is a research/scanner script. It does not place
# orders and does not output position sizing. Treat every alert as
# something to verify yourself before acting on it.
# ============================================================

from __future__ import annotations

import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================= CONFIG =========================
KRAKEN_API_URL = "https://api.kraken.com/0/public"

# Kraken's OHLC endpoint takes interval in minutes.
KRAKEN_INTERVAL_MAP = {
    "4h": 240,
    "15m": 15,
    "5m": 5,
    "1d": 1440,
    "1w": 10080,
}

# Populated at runtime by get_valid_futures_symbols(): maps our display
# symbol (e.g. "BTCUSDT") to the pair code Kraken actually expects
# (e.g. "XBTUSDT", since Kraken uses "XBT" instead of "BTC").
_KRAKEN_PAIR_MAP: dict[str, str] = {}

DATA_LOOKBACK_4H = 300
DATA_LOOKBACK_15M = 300
DATA_LOOKBACK_5M = 200
DATA_LOOKBACK_1D = 250
DATA_LOOKBACK_1W = 120

MAX_WORKERS = 8
HTTP_TIMEOUT = 12
HTTP_MIN_INTERVAL = 0.07
CACHE_TTL_SECONDS = 25

MIN_VOLUME_USDT = 8_000_000
MIN_ATR_SCALP = 0.65

SWING_PROM_ATR = 0.55
SWING_DISTANCE = 4
MAJOR_SWING_PROM_ATR = 0.95
MAJOR_SWING_DISTANCE = 7

CONSOL_MIN_CANDLES = 5
CONSOL_MAX_CANDLES = 28
CONSOL_RANGE_ATR_MULT = 1.35
BREAKOUT_BODY_MULT = 1.45
BREAKOUT_VOL_MULT = 1.25

FVG_MAX_AGE_BARS = 70
FVG_MAX_DISTANCE_ATR = 2.8
FVG_FULL_FILL_PCT = 0.97
OB_LOOKBACK = 70
OB_DISPLACEMENT_ATR = 1.15
OB_MAX_DISTANCE_ATR = 2.2
OB_MIN_IMPULSE_BARS = 2

MSS_LOOKBACK = 8
ENTRY_DISPLACEMENT_ATR = 0.38
ENTRY_CLOSE_OUTER_THIRD = 0.66

# Universe: a small static watchlist + top-N by Kraken 24h volume
ALWAYS_INCLUDE_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT",
]
STABLE_BLACKLIST = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD",
    "USDS", "USD1", "BUSD", "USDP",
}
TOP_N_BY_VOLUME = 50
ALERT_TIERS = {"A+", "A"}          # only these tiers trigger a Telegram message
ALERT_DEDUP_HOURS = 20             # don't re-alert the same setup within this window

STATE_DIR = Path(os.environ.get("STATE_DIR", "state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
ALERTED_FILE = STATE_DIR / "alerted.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ========================= HTTP / CACHE =========================
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Crypto-Alert-Bot/1.0"})
adapter = HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=Retry(total=0, redirect=0))
_SESSION.mount("https://", adapter)
_SESSION.mount("http://", adapter)

_HTTP_LOCK = threading.Lock()
_LAST_HTTP_TS = 0.0
_API_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_LOCK = threading.Lock()
_INVALID_SYMBOLS: set[str] = set()
_INVALID_LOCK = threading.Lock()


def _rate_limited_get(url: str, timeout: int = HTTP_TIMEOUT):
    global _LAST_HTTP_TS
    with _HTTP_LOCK:
        elapsed = time.monotonic() - _LAST_HTTP_TS
        if elapsed < HTTP_MIN_INTERVAL:
            time.sleep(HTTP_MIN_INTERVAL - elapsed)
        _LAST_HTTP_TS = time.monotonic()
    return _SESSION.get(url, timeout=timeout)


def make_api_request(url: str, retries: int = 3, use_cache: bool = True, ttl: int = CACHE_TTL_SECONDS):
    now = time.time()
    if use_cache:
        with _CACHE_LOCK:
            hit = _API_CACHE.get(url)
            if hit and (now - hit[0]) < ttl:
                return hit[1]

    if "symbol=" in url:
        sym = url.split("symbol=")[1].split("&")[0]
        with _INVALID_LOCK:
            if sym in _INVALID_SYMBOLS:
                return None

    backoff = 0.6
    for attempt in range(retries):
        try:
            r = _rate_limited_get(url)
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", backoff))
                time.sleep(min(max(retry_after, backoff), 12.0))
                backoff *= 2.0
                continue
            if r.status_code == 400:
                if "symbol=" in url:
                    sym = url.split("symbol=")[1].split("&")[0]
                    with _INVALID_LOCK:
                        _INVALID_SYMBOLS.add(sym)
                return None
            if 500 <= r.status_code < 600:
                raise requests.HTTPError(f"server status {r.status_code}")
            r.raise_for_status()
            data = r.json()
            if use_cache:
                with _CACHE_LOCK:
                    _API_CACHE[url] = (time.time(), data)
            return data
        except Exception as exc:
            if attempt == retries - 1:
                logging.warning("API failed after %s attempts: %s | %s", retries, url, exc)
                return None
            time.sleep(backoff)
            backoff *= 1.6
    return None


def get_valid_futures_symbols() -> set[str]:
    """Kraken equivalent of Binance's exchangeInfo: list all tradeable
    USDT-quoted pairs. Builds _KRAKEN_PAIR_MAP as a side effect, since
    Kraken's own pair codes (e.g. "XBTUSDT" for Bitcoin) don't always
    match the plain "BTCUSDT"-style symbol we use elsewhere in the script."""
    global _KRAKEN_PAIR_MAP
    data = make_api_request(f"{KRAKEN_API_URL}/AssetPairs", ttl=3600)
    if not data or data.get("error"):
        logging.warning("Could not fetch Kraken AssetPairs — falling back to empty valid set")
        return set()
    result = data.get("result", {})
    valid = set()
    mapping: dict[str, str] = {}
    for pair_key, info in result.items():
        quote = info.get("quote", "")
        if quote != "USDT":
            continue
        base = info.get("base", "")
        display_base = base
        # Kraken prefixes some legacy assets with "X" (e.g. "XXBT", "XETH")
        if display_base.startswith("X") and len(display_base) == 4:
            display_base = display_base[1:]
        if display_base == "XBT":
            display_base = "BTC"
        our_symbol = f"{display_base}USDT"
        api_pair = info.get("altname", pair_key)
        valid.add(our_symbol)
        mapping[our_symbol] = api_pair
    _KRAKEN_PAIR_MAP = mapping
    return valid


def get_top_symbols_by_volume(valid_futures: set[str], top_n: int = TOP_N_BY_VOLUME) -> list[str]:
    """Replaces the old CoinGecko market-cap ranking: rank Kraken USDT
    pairs by approximate 24h quote volume (base volume x VWAP), using
    data we already have access to."""
    if not _KRAKEN_PAIR_MAP:
        return []
    pairs = [(sym, code) for sym, code in _KRAKEN_PAIR_MAP.items()
              if sym[:-4] not in STABLE_BLACKLIST]
    rows = []
    chunk_size = 20
    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i:i + chunk_size]
        pair_codes = ",".join(code for _, code in chunk)
        url = f"{KRAKEN_API_URL}/Ticker?pair={pair_codes}"
        data = make_api_request(url, ttl=300, use_cache=True)
        if not data or data.get("error"):
            continue
        result = data.get("result", {})
        reverse_map = {code: sym for sym, code in chunk}
        for key, info in result.items():
            sym = reverse_map.get(key)
            if sym is None:
                continue
            try:
                vol24 = float(info["v"][1])
                vwap24 = float(info["p"][1])
                qv = vol24 * vwap24
            except (KeyError, TypeError, ValueError, IndexError):
                qv = 0.0
            rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in rows[:top_n]]


def fetch_kline(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame | None:
    with _INVALID_LOCK:
        if symbol in _INVALID_SYMBOLS:
            return None
    interval_min = KRAKEN_INTERVAL_MAP.get(interval)
    if interval_min is None:
        logging.warning("Unknown interval %s — no Kraken mapping", interval)
        return None
    kraken_pair = _KRAKEN_PAIR_MAP.get(symbol)
    if not kraken_pair:
        # Fallback guess if the symbol map hasn't been populated yet
        base = symbol[:-4]
        kraken_pair = ("XBTUSDT" if base == "BTC" else f"{base}USDT")
    url = f"{KRAKEN_API_URL}/OHLC?pair={kraken_pair}&interval={interval_min}"
    data = make_api_request(url, use_cache=False)
    if not data or data.get("error"):
        with _INVALID_LOCK:
            _INVALID_SYMBOLS.add(symbol)
        return None
    result = data.get("result", {})
    series = None
    for k, v in result.items():
        if k == "last":
            continue
        series = v
        break
    if not series:
        return None
    rows = series[-limit:] if limit else series
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "vwap", "volume", "count"])
    numeric = ["open", "high", "low", "close", "vwap", "volume"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="s", utc=True).astype("datetime64[ns, UTC]")
    df["quote_volume"] = df["volume"] * df["close"]  # approx quote volume
    df["close_time"] = df["open_time"].shift(-1)
    df.loc[df.index[-1], "close_time"] = pd.Timestamp.now(tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["open_time"] <= now].copy()
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    df.rename(columns={"open_time": "time"}, inplace=True)
    # buy_vol (taker-buy volume) isn't in this endpoint; not used by any
    # active signal in this script, so fill with a neutral placeholder.
    df["buy_vol"] = df["volume"] / 2
    return df[["time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "buy_vol"]]


# ========================= INDICATOR HELPERS =========================
def safe_float(v, default=None):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def ema_series(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def atr_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def adx_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = atr_series(df, length)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def latest_atr(df: pd.DataFrame, length: int = 14) -> float | None:
    s = atr_series(df, length)
    if s is None or s.dropna().empty:
        return None
    return safe_float(s.dropna().iloc[-1])


def distance_atr(price: float, level: float | None, atr: float | None) -> float | None:
    if level is None or atr is None or atr <= 0:
        return None
    return abs(price - level) / atr


def candle_close_position(row: pd.Series) -> float:
    rng = row["high"] - row["low"]
    if rng <= 0:
        return 0.5
    return (row["close"] - row["low"]) / rng


def find_swings(df: pd.DataFrame, major: bool = False) -> tuple[np.ndarray, np.ndarray]:
    atr = latest_atr(df)
    if atr is None or atr <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    prom = (MAJOR_SWING_PROM_ATR if major else SWING_PROM_ATR) * atr
    dist = MAJOR_SWING_DISTANCE if major else SWING_DISTANCE
    hi, _ = find_peaks(df["high"].to_numpy(), distance=dist, prominence=prom)
    lo, _ = find_peaks(-df["low"].to_numpy(), distance=dist, prominence=prom)
    return hi, lo


# ========================= TREND / STRUCTURE =========================
def confirm_trend(df: pd.DataFrame | None, label: str = "4H") -> tuple[str, float]:
    if df is None or len(df) < 60:
        return f"Unknown ({label})", 0.0
    e9 = ema_series(df["close"], 9)
    e21 = ema_series(df["close"], 21)
    e50 = ema_series(df["close"], 50)
    c = df["close"].iloc[-1]
    score = 0
    if c > e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]:
        score += 3
    elif c < e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]:
        score -= 3
    elif c > e21.iloc[-1] > e50.iloc[-1]:
        score += 2
    elif c < e21.iloc[-1] < e50.iloc[-1]:
        score -= 2
    h1 = df["high"].iloc[-15:].max()
    h0 = df["high"].iloc[-30:-15].max()
    l1 = df["low"].iloc[-15:].min()
    l0 = df["low"].iloc[-30:-15].min()
    if h1 > h0 and l1 > l0:
        score += 2
    elif h1 < h0 and l1 < l0:
        score -= 2
    if score >= 5:
        return f"Strong Uptrend ({label})", 1.0
    if score >= 2:
        return f"Uptrend ({label})", 0.6
    if score <= -5:
        return f"Strong Downtrend ({label})", -1.0
    if score <= -2:
        return f"Downtrend ({label})", -0.6
    return f"Ranging ({label})", 0.0


def detect_regime(df4h: pd.DataFrame) -> tuple[str, float, str]:
    if df4h is None or len(df4h) < 80:
        return "UNKNOWN", 0.0, "Insufficient data"
    d = df4h.copy()
    ema21 = ema_series(d["close"], 21)
    ema50 = ema_series(d["close"], 50)
    adx_s = adx_series(d, 14)
    atr = latest_atr(d)
    if atr is None or adx_s.dropna().empty:
        return "UNKNOWN", 0.0, "Indicator failure"
    adx = safe_float(adx_s.dropna().iloc[-1], 0.0)
    close = d["close"].iloc[-1]
    ema21v = ema21.iloc[-1]
    ema50v = ema50.iloc[-1]
    slope_look = min(8, len(d) - 2)
    e21_slope = (ema21.iloc[-1] - ema21.iloc[-1 - slope_look]) / atr
    range_width = (d["high"].iloc[-30:].max() - d["low"].iloc[-30:].min()) / atr
    if adx >= 25 and close > ema21v > ema50v and e21_slope > 0.45:
        return "TREND_UP", min(1.0, 0.5 + adx / 100), f"ADX={adx:.1f} | EMA stack up"
    if adx >= 25 and close < ema21v < ema50v and e21_slope < -0.45:
        return "TREND_DOWN", min(1.0, 0.5 + adx / 100), f"ADX={adx:.1f} | EMA stack down"
    if range_width < 7.5 and adx < 20:
        return "RANGE", 0.5, f"ADX={adx:.1f} | 30-bar width={range_width:.1f} ATR"
    if atr / close * 100 < MIN_ATR_SCALP:
        return "LOW_VOL", 0.35, f"ATR%={atr / close * 100:.2f}"
    return "TRANSITION", 0.4, f"ADX={adx:.1f} | mixed"


def detect_15m_swing_structure(df: pd.DataFrame) -> dict:
    out = {"label": "Insufficient", "score": 0.0, "last_hl": None, "last_lh": None}
    if df is None or len(df) < 80:
        return out
    hi_idx, lo_idx = find_swings(df)
    if len(hi_idx) < 3 or len(lo_idx) < 3:
        out["label"] = "Not Enough Swings"
        return out
    highs = [float(df["high"].iloc[i]) for i in hi_idx[-3:]]
    lows = [float(df["low"].iloc[i]) for i in lo_idx[-3:]]
    out["last_hl"] = lows[-1]
    out["last_lh"] = highs[-1]
    hh = highs[-1] > highs[-2] > highs[-3]
    hl = lows[-1] > lows[-2] > lows[-3]
    lh = highs[-1] < highs[-2] < highs[-3]
    ll = lows[-1] < lows[-2] < lows[-3]
    if hh and hl:
        out["label"] = "HH + HL Bullish"
        out["score"] = 1.0
    elif lh and ll:
        out["label"] = "LH + LL Bearish"
        out["score"] = -1.0
    elif hh:
        out["label"] = "HH Only"
        out["score"] = 0.35
    elif ll:
        out["label"] = "LL Only"
        out["score"] = -0.35
    else:
        out["label"] = "Choppy"
    return out


# ========================= FVG / ORDER BLOCK / SWEEP / BREAKOUT =========================
def detect_fair_value_gaps(df: pd.DataFrame, label: str = "4H") -> dict:
    out = {"label": "No FVG", "direction": 0, "level": None, "age": None, "fill_pct": None, "fresh": False, "high": None, "low": None}
    if df is None or len(df) < 10:
        return out
    atr = latest_atr(df)
    if atr is None or atr <= 0:
        return out
    price = float(df["close"].iloc[-1])
    candidates = []
    start = max(2, len(df) - FVG_MAX_AGE_BARS)
    for i in range(start, len(df)):
        a = df.iloc[i - 2]
        c = df.iloc[i]
        if c["low"] > a["high"]:
            low, high = float(a["high"]), float(c["low"])
            mid = (low + high) / 2
            if price <= high + FVG_MAX_DISTANCE_ATR * atr:
                candidates.append((abs(price - mid) / atr, i, 1, low, high, mid))
        if c["high"] < a["low"]:
            low, high = float(c["high"]), float(a["low"])
            mid = (low + high) / 2
            if price >= low - FVG_MAX_DISTANCE_ATR * atr:
                candidates.append((abs(price - mid) / atr, i, -1, low, high, mid))
    if not candidates:
        return out
    candidates.sort(key=lambda x: x[0])
    _, idx, direction, low, high, mid = candidates[0]
    later = df.iloc[idx + 1:]
    fill_pct = 0.0
    if len(later):
        if direction > 0:
            deepest = float(later["low"].min())
            fill_pct = np.clip((high - deepest) / max(high - low, 1e-12), 0, 1)
        else:
            highest = float(later["high"].max())
            fill_pct = np.clip((highest - low) / max(high - low, 1e-12), 0, 1)
    if fill_pct >= FVG_FULL_FILL_PCT:
        return out
    age = len(df) - 1 - idx
    out.update({
        "label": f"{'Bullish' if direction > 0 else 'Bearish'} FVG {label} {low:.6f}-{high:.6f}",
        "direction": direction, "level": mid, "age": age,
        "fill_pct": round(float(fill_pct) * 100, 1),
        "fresh": age <= 8 and fill_pct < 0.22, "high": high, "low": low,
    })
    return out


def detect_order_blocks(df: pd.DataFrame, close: float) -> dict:
    out = {"label": "No Active OB", "direction": 0, "mid": None, "low": None, "high": None, "fresh": False, "age": None}
    if df is None or len(df) < 45:
        return out
    atr = latest_atr(df)
    if atr is None or atr <= 0:
        return out
    candidates = []
    start = max(3, len(df) - OB_LOOKBACK)
    for i in range(start, len(df) - OB_MIN_IMPULSE_BARS - 1):
        row = df.iloc[i]
        future = df.iloc[i + 1: i + 1 + OB_MIN_IMPULSE_BARS + 2]
        if len(future) < OB_MIN_IMPULSE_BARS:
            continue
        if row["close"] < row["open"]:
            impulse_high = float(future["high"].max())
            move = (impulse_high - float(row["high"])) / atr
            if move >= OB_DISPLACEMENT_ATR and impulse_high > float(row["high"]):
                candidates.append((i, 1, float(row["low"]), float(row["high"])))
        elif row["close"] > row["open"]:
            impulse_low = float(future["low"].min())
            move = (float(row["low"]) - impulse_low) / atr
            if move >= OB_DISPLACEMENT_ATR and impulse_low < float(row["low"]):
                candidates.append((i, -1, float(row["low"]), float(row["high"])))
    if not candidates:
        return out
    relevant = []
    for i, direction, low, high in reversed(candidates):
        mid = (low + high) / 2
        dist = distance_atr(close, mid, atr)
        if dist is None or dist > OB_MAX_DISTANCE_ATR:
            continue
        later = df.iloc[i + 1:]
        violated = (bool((later["close"] < low).any()) if direction > 0
                    else bool((later["close"] > high).any())) if len(later) else False
        relevant.append((dist, i, direction, low, high, not violated))
    if not relevant:
        return out
    relevant.sort(key=lambda x: (x[0], -x[1]))
    _, idx, direction, low, high, fresh = relevant[0]
    age = len(df) - 1 - idx
    out.update({
        "label": f"{'Bullish' if direction > 0 else 'Bearish'} OB {low:.6f}-{high:.6f}{' Fresh' if fresh else ' Retested'}",
        "direction": direction, "mid": (low + high) / 2, "low": low, "high": high,
        "fresh": bool(fresh), "age": age,
    })
    return out


def detect_liquidity_sweep(df: pd.DataFrame, label: str = "15M", lookback_bars: int = 5) -> dict:
    out = {"label": "No Sweep", "direction": 0, "level": None, "age": None, "strength": 0.0}
    if df is None or len(df) < 70:
        return out
    atr = latest_atr(df)
    if atr is None or atr <= 0:
        return out
    hi_idx, lo_idx = find_swings(df)
    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return out

    def recent_cluster_level(idxs, is_high: bool, tol_atr: float = 0.35):
        if len(idxs) < 2:
            return None
        recent = idxs[-4:]
        levels = [float(df["high" if is_high else "low"].iloc[i]) for i in recent]
        for i in range(len(levels) - 1, 0, -1):
            for j in range(i):
                if abs(levels[i] - levels[j]) / atr <= tol_atr:
                    return levels[i]
        return levels[-1]

    prior_high = recent_cluster_level(hi_idx, True)
    prior_low = recent_cluster_level(lo_idx, False)

    for bars_ago in range(lookback_bars):
        idx = len(df) - 1 - bars_ago
        if idx < 3:
            break
        bar = df.iloc[idx]
        if prior_low is not None and bar["low"] < prior_low and bar["close"] > prior_low:
            wick = (prior_low - bar["low"]) / atr
            if wick >= 0.15:
                out.update({
                    "label": f"Bullish Sweep {label} below {prior_low:.6f}",
                    "direction": 1, "level": prior_low, "age": bars_ago,
                    "strength": min(1.0, 0.45 + wick / 1.8),
                })
                return out
        if prior_high is not None and bar["high"] > prior_high and bar["close"] < prior_high:
            wick = (bar["high"] - prior_high) / atr
            if wick >= 0.15:
                out.update({
                    "label": f"Bearish Sweep {label} above {prior_high:.6f}",
                    "direction": -1, "level": prior_high, "age": bars_ago,
                    "strength": min(1.0, 0.45 + wick / 1.8),
                })
                return out
    return out


def detect_consolidation_breakout(df: pd.DataFrame) -> dict:
    out = {"label": "No Breakout", "direction": 0, "strength": 0.0, "res": None, "sup": None}
    if df is None or len(df) < CONSOL_MIN_CANDLES + 10:
        return out
    atr = latest_atr(df)
    if atr is None or atr <= 0:
        return out
    bo = df.iloc[-1]
    base = df.iloc[-(CONSOL_MAX_CANDLES + 1):-1]
    best = None
    best_rng = np.inf
    for end in range(len(base), CONSOL_MIN_CANDLES - 1, -1):
        for start in range(max(0, end - CONSOL_MAX_CANDLES), end - CONSOL_MIN_CANDLES + 1):
            sub = base.iloc[start:end]
            rng = float(sub["high"].max() - sub["low"].min())
            if rng <= CONSOL_RANGE_ATR_MULT * atr and rng < best_rng:
                best = sub
                best_rng = rng
        if best is not None:
            break
    if best is None:
        return out
    res = float(best["high"].max())
    sup = float(best["low"].min())
    body = abs(float(bo["close"] - bo["open"]))
    median_body = float((best["close"] - best["open"]).abs().median()) or atr * 0.1
    body_ratio = body / median_body
    vol_ma = float(df["volume"].iloc[-21:-1].mean())
    vol_ratio = float(bo["volume"] / vol_ma) if vol_ma > 0 else 1.0
    bull = bo["close"] > res and bo["close"] > bo["open"]
    bear = bo["close"] < sup and bo["close"] < bo["open"]
    if not bull and not bear:
        out.update({"res": res, "sup": sup})
        return out
    strength = 0.25
    if body_ratio >= BREAKOUT_BODY_MULT:
        strength += 0.30
    if vol_ratio >= BREAKOUT_VOL_MULT:
        strength += 0.25
    if best_rng / atr <= 1.0:
        strength += 0.15
    direction = 1 if bull else -1
    out.update({
        "label": f"{'Bullish' if bull else 'Bearish'} Breakout {len(best)}c",
        "direction": direction, "strength": min(1.0, strength), "res": res, "sup": sup,
    })
    return out


def check_mtf_alignment(weekly, daily, four_h, direction: int) -> dict:
    out = {"veto": False, "full": False, "score": 0.0, "label": "No direction"}
    if direction == 0:
        return out
    values = [weekly[1], daily[1], four_h[1]]
    signed = [v if direction > 0 else -v for v in values]
    if signed[0] <= -0.5:
        out.update({"veto": True, "label": "MTF VETO — Weekly trend opposes setup"})
        return out
    score = sum(max(v, 0) for v in signed)
    full = all(v >= 0.5 for v in signed)
    details = [f"W={weekly[0]}", f"D={daily[0]}", f"4H={four_h[0]}"]
    out.update({
        "score": round(score, 2), "full": full,
        "label": ("FULL MTF ALIGNMENT | " if full else "PARTIAL MTF | ") + " | ".join(details),
    })
    return out


def detect_5m_entry(df5m: pd.DataFrame, direction: int) -> dict:
    out = {"label": "WAIT", "confirmed": False, "strength": 0.0}
    if df5m is None or len(df5m) < 60 or direction == 0:
        return out
    df = df5m.copy()
    atr = latest_atr(df)
    if atr is None or atr <= 0:
        return out
    e9 = ema_series(df["close"], 9)
    e21 = ema_series(df["close"], 21)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    now = df.iloc[-1]
    prior = df.iloc[-MSS_LOOKBACK - 1:-1]
    body = abs(float(now["close"] - now["open"]))
    displacement = body / atr >= ENTRY_DISPLACEMENT_ATR
    close_pos = candle_close_position(now)
    if direction > 0:
        mss = float(now["close"]) > float(prior["high"].max())
        vwap_ok = float(now["close"]) > float(vwap.iloc[-1])
        ema_ok = float(now["close"]) > float(e9.iloc[-1]) > float(e21.iloc[-1])
        candle_ok = close_pos >= ENTRY_CLOSE_OUTER_THIRD
    else:
        mss = float(now["close"]) < float(prior["low"].min())
        vwap_ok = float(now["close"]) < float(vwap.iloc[-1])
        ema_ok = float(now["close"]) < float(e9.iloc[-1]) < float(e21.iloc[-1])
        candle_ok = close_pos <= (1 - ENTRY_CLOSE_OUTER_THIRD)
    score = sum([mss, displacement, vwap_ok, ema_ok, candle_ok]) / 5.0
    out.update({
        "label": f"{'LONG' if direction > 0 else 'SHORT'} {'CONFIRMED' if score >= 0.8 else 'WATCH'}",
        "confirmed": bool(score >= 0.8), "strength": round(float(score), 2),
    })
    return out


# ========================= SCORING =========================
def choose_primary_direction(evidence: dict) -> tuple[int, float, str]:
    long_core = short_core = 0.0
    for item in evidence.values():
        d = item.get("direction", 0)
        s = float(item.get("strength", 0.0))
        w = float(item.get("weight", 1.0))
        contribution = float(np.clip(s * w, 0, 1.8))
        if d > 0:
            long_core += contribution
        elif d < 0:
            short_core += contribution
    long_core, short_core = min(long_core, 9.5), min(short_core, 9.5)
    diff = long_core - short_core
    direction = 1 if diff >= 1.85 else -1 if diff <= -1.85 else 0
    label = (f"Long evidence={long_core:.2f}" if direction > 0
             else f"Short evidence={short_core:.2f}" if direction < 0
             else f"Conflicted/weak | L={long_core:.2f} S={short_core:.2f}")
    return direction, round(diff, 2), label


def setup_type(sweep: dict, ob: dict, fvg: dict, bo: dict, structure: dict) -> str:
    if sweep["direction"] and ob["direction"] == sweep["direction"] and ob["fresh"]:
        return "LIQUIDITY_REVERSAL_OB"
    if sweep["direction"]:
        return "LIQUIDITY_REVERSAL"
    if ob["direction"] and ob["fresh"]:
        return "ORDER_BLOCK_RETEST"
    if fvg["direction"] and fvg["fresh"]:
        return "FVG_REVERSION"
    if bo["direction"]:
        return "BREAKOUT"
    if structure["score"] != 0:
        return "TREND_CONTINUATION"
    return "NO_SETUP"


def build_quality(direction: int, regime: str, mtf: dict, entry: dict, setup: str) -> dict:
    if direction == 0:
        return {"score": 0.0, "tier": "C", "label": "NO TRADE"}
    components = [
        0.25 * (1.0 if mtf.get("full") else 0.45 if not mtf.get("veto") else 0.0),
        0.25 * float(entry.get("strength", 0.0)),
    ]
    setup_bonus = {
        "LIQUIDITY_REVERSAL_OB": 1.0, "LIQUIDITY_REVERSAL": 0.82,
        "ORDER_BLOCK_RETEST": 0.78, "FVG_REVERSION": 0.62,
        "BREAKOUT": 0.68, "TREND_CONTINUATION": 0.55, "NO_SETUP": 0.0,
    }.get(setup, 0.0)
    components.append(0.25 * setup_bonus)
    regime_bonus = 1.0 if regime in ("TREND_UP", "TREND_DOWN") else 0.60 if regime == "TRANSITION" else 0.40
    components.append(0.25 * regime_bonus)
    score = round(float(np.clip(sum(components), 0, 1)) * 100, 1)
    if mtf.get("veto"):
        tier = "C"
    elif score >= 83 and entry.get("confirmed") and mtf.get("full"):
        tier = "A+"
    elif score >= 73 and (entry.get("confirmed") or mtf.get("full")):
        tier = "A"
    elif score >= 58:
        tier = "B"
    else:
        tier = "C"
    label = {"A+": "QUALIFIED", "A": "HIGH QUALITY", "B": "WATCHLIST", "C": "NO TRADE"}[tier]
    return {"score": score, "tier": tier, "label": label}


# ========================= ANALYSIS =========================
def analyze_symbol(symbol: str, df_btc_4h: pd.DataFrame | None = None) -> dict | None:
    try:
        df4h = df_btc_4h if symbol == "BTCUSDT" and df_btc_4h is not None else fetch_kline(symbol, "4h", DATA_LOOKBACK_4H)
        df15m = fetch_kline(symbol, "15m", DATA_LOOKBACK_15M)
        df5m = fetch_kline(symbol, "5m", DATA_LOOKBACK_5M)
        df_d = fetch_kline(symbol, "1d", DATA_LOOKBACK_1D)
        df_w = fetch_kline(symbol, "1w", DATA_LOOKBACK_1W)
        if any(x is None for x in (df4h, df15m, df5m, df_d, df_w)) or len(df4h) < 100:
            return None

        close = float(df4h["close"].iloc[-1])
        vol_24h = float(df4h["quote_volume"].iloc[-6:].sum())
        if vol_24h < MIN_VOLUME_USDT:
            return None

        trend_w = confirm_trend(df_w, "Weekly")
        trend_d = confirm_trend(df_d, "Daily")
        trend_4h = confirm_trend(df4h, "4H")
        regime, regime_strength, regime_detail = detect_regime(df4h)
        structure = detect_15m_swing_structure(df15m)
        sweep = detect_liquidity_sweep(df15m)
        ob = detect_order_blocks(df4h, close)
        fvg = detect_fair_value_gaps(df4h)
        bo = detect_consolidation_breakout(df15m)

        evidence = {
            "trend": {"direction": 1 if trend_4h[1] > 0 else -1 if trend_4h[1] < 0 else 0, "strength": abs(trend_4h[1]), "weight": 1.7},
            "structure": {"direction": 1 if structure["score"] > 0 else -1 if structure["score"] < 0 else 0, "strength": abs(structure["score"]), "weight": 1.45},
            "sweep": {"direction": sweep["direction"], "strength": sweep["strength"], "weight": 2.05},
            "ob": {"direction": ob["direction"], "strength": 1.0 if ob["fresh"] else 0.50, "weight": 1.65},
            "fvg": {"direction": fvg["direction"], "strength": 0.88 if fvg["fresh"] else 0.42, "weight": 1.05},
            "breakout": {"direction": bo["direction"], "strength": bo["strength"], "weight": 1.35},
        }
        direction, edge_score, edge_label = choose_primary_direction(evidence)
        setup = setup_type(sweep, ob, fvg, bo, structure)
        mtf = check_mtf_alignment(trend_w, trend_d, trend_4h, direction)
        entry = detect_5m_entry(df5m, direction) if direction != 0 else {"label": "WAIT", "confirmed": False, "strength": 0.0}
        quality = build_quality(direction, regime, mtf, entry, setup)

        atr = latest_atr(df4h) or 0.0
        atr_pct = atr / close * 100 if close else 0.0

        confluences = []
        if mtf["full"]:
            confluences.append("MTF")
        if sweep["direction"] == direction and direction:
            confluences.append("Sweep")
        if ob["direction"] == direction and direction:
            confluences.append("Fresh OB" if ob["fresh"] else "OB")
        if fvg["direction"] == direction and direction:
            confluences.append("FVG")
        if bo["direction"] == direction and direction:
            confluences.append("Breakout")
        if structure["score"] * direction > 0.5:
            confluences.append("Structure")
        if entry.get("confirmed"):
            confluences.append("5M Entry")

        invalidation = None
        if direction > 0:
            refs = [x for x in [structure.get("last_hl"), ob.get("low")] if x is not None and x < close]
            invalidation = round((close - max(refs)) / close * 100, 2) if refs else None
        elif direction < 0:
            refs = [x for x in [structure.get("last_lh"), ob.get("high")] if x is not None and x > close]
            invalidation = round((min(refs) - close) / close * 100, 2) if refs else None

        bias = ("LONG VETO" if mtf["veto"] and direction > 0 else
                "SHORT VETO" if mtf["veto"] and direction < 0 else
                "Strong Long" if direction > 0 and quality["tier"] == "A+" else
                "Long Bias" if direction > 0 else
                "Strong Short" if direction < 0 and quality["tier"] == "A+" else
                "Short Bias" if direction < 0 else "Neutral")

        return {
            "Symbol": symbol,
            "Close": close,
            "Direction": direction,
            "Bias": bias,
            "Tier": quality["tier"],
            "Quality_Score": quality["score"],
            "Regime": regime,
            "Setup_Type": setup,
            "MTF_Veto": "Yes" if mtf["veto"] else "No",
            "MTF_Alignment": mtf["label"],
            "Sweep": sweep["label"],
            "Order_Block": ob["label"],
            "FVG": fvg["label"],
            "Breakout": bo["label"],
            "5M_Entry": entry["label"],
            "5M_Confirmed": "Yes" if entry.get("confirmed") else "No",
            "ATR_%": round(atr_pct, 3),
            "Invalidation_Dist_%": invalidation,
            "Confluence_Detail": " + ".join(confluences) if confluences else "None",
        }
    except Exception as exc:
        logging.warning("Error analyzing %s: %s", symbol, exc)
        return None


# ========================= TELEGRAM =========================
def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured — skipping send. Set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        logging.error("Telegram send failed: %s", exc)
        return False


def format_alert(row: dict) -> str:
    return (
        f"*{row['Symbol']}* — {row['Bias']}  (Tier {row['Tier']}, Score {row['Quality_Score']})\n"
        f"Setup: {row['Setup_Type'].replace('_', ' ').title()}\n"
        f"{row['MTF_Alignment']}\n"
        f"Sweep: {row['Sweep']}\n"
        f"Order Block: {row['Order_Block']}\n"
        f"5M Entry: {row['5M_Entry']}\n"
        f"Confluences: {row['Confluence_Detail']}\n"
        f"Invalidation: ~{row['Invalidation_Dist_%']}% away\n"
        f"Price: {row['Close']}"
    )


# ========================= DE-DUP STATE =========================
def load_alerted_state() -> dict:
    if ALERTED_FILE.exists():
        try:
            return json.loads(ALERTED_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_alerted_state(state: dict) -> None:
    ALERTED_FILE.write_text(json.dumps(state, indent=2))


def signal_key(row: dict) -> str:
    return f"{row['Symbol']}|{row['Setup_Type']}|{row['Direction']}|{row['Tier']}"


def prune_state(state: dict) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=ALERT_DEDUP_HOURS)
    return {
        k: v for k, v in state.items()
        if datetime.fromisoformat(v) > cutoff
    }


# ========================= MAIN =========================
def run_scan() -> pd.DataFrame:
    logging.info("Fetching valid Kraken USDT pairs...")
    valid_futures = get_valid_futures_symbols()

    logging.info("Fetching BTC reference data...")
    df_btc = fetch_kline("BTCUSDT", "4h", DATA_LOOKBACK_4H)
    if df_btc is None or len(df_btc) < 100:
        logging.error("BTC reference data unavailable — aborting this run")
        return pd.DataFrame()

    top_by_volume = get_top_symbols_by_volume(valid_futures, TOP_N_BY_VOLUME)
    symbols = list(dict.fromkeys(ALWAYS_INCLUDE_SYMBOLS + top_by_volume))
    symbols = [s for s in symbols if s in valid_futures]
    logging.info("Scanning %d symbols", len(symbols))

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(analyze_symbol, s, df_btc): s for s in symbols}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as exc:
                logging.warning("%s failed: %s", futs[f], exc)
                r = None
            if r:
                results.append(r)

    df_main = pd.DataFrame(results)
    if df_main.empty:
        return df_main
    return df_main.sort_values(["Quality_Score"], ascending=False).reset_index(drop=True)


def main():
    df_main = run_scan()
    if df_main.empty:
        logging.info("No results this run.")
        return

    state = prune_state(load_alerted_state())
    sent = 0
    for _, row in df_main.iterrows():
        row = row.to_dict()
        if row.get("MTF_Veto") == "Yes":
            continue
        if row.get("Tier") not in ALERT_TIERS:
            continue
        key = signal_key(row)
        if key in state:
            continue  # already alerted recently — skip to avoid spam
        message = format_alert(row)
        if send_telegram_message(message):
            state[key] = datetime.now(timezone.utc).isoformat()
            sent += 1

    save_alerted_state(state)
    top10 = df_main.head(10)[["Symbol", "Bias", "Tier", "Quality_Score", "Setup_Type", "Confluence_Detail"]]
    logging.info("Scan complete. %d symbols analyzed, %d new alerts sent.", len(df_main), sent)
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
