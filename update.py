#!/usr/bin/env python3
"""
Krypto-Dashboard v4 – läuft automatisch über GitHub Actions (stündlich).

Ablauf:
1. Top-Coins nach Marktkapitalisierung (CoinGecko, Ersatz CoinPaprika), ohne Stablecoins/Wrapped.
2. Kerzen in 9 Zeitrahmen (Binance, Ersatz OKX/KuCoin). Tag/Woche/Monat mit kompletter Historie.
3. Markt-Daten: Fear & Greed Index, Dollar-Index (DXY), Gold, BTC-Dominanz (Näherung),
   eigener Gesamtmarkt-Index aus den Top-20-Coins.
4. Antizyklischer Score 0–100 je Kerze (ohne Repainting) aus Lage + Markt (Richtung),
   Wende (Bestätigung), übergeordnetem Zeitrahmen und Gesamtmarkt (Filter).
5. Wöchentliche Auto-Kalibrierung der Gewichte mit Trainings- und Testzeitraum.
6. Schreibt site/index.html, site/data/<COIN>.json, site/meta.json, site/calib.json.

Keine Anlageberatung.
"""
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------- Einstellungen
TOP_N = 100
CANDIDATES = 250
INTRADAY_KEEP = 1000   # 1 Std / 4 Std: so viele Kerzen vom Server (ältere lädt das Handy nach)
SCORE_MIN = 80
BUY_LEVEL, SELL_LEVEL = 80, 20
ZONE_SHARE = 0.04      # Anteil der Kerzen, die ungefähr in einer Zone landen sollen
PIVOT_N = 10
STAT_HORIZON = 10
INDEX_COINS = 20
CALIB_MAX_AGE_DAYS = 7
CALIB_COINS = 40
OUT_DIR = "site"

LIST_TFS = ["1h", "4h", "1d", "1w"]
TIMEFRAMES = [("1h", "1 Std"), ("4h", "4 Std"), ("1d", "1 T"), ("5d", "5 T"), ("1w", "1 W"),
              ("2w", "2 W"), ("1M", "1 M"), ("6M", "6 M"), ("1y", "1 J")]
TF_SEC = {"1h": 3600, "4h": 14400, "1d": 86400, "5d": 432000, "1w": 604800, "2w": 1209600,
          "1M": 2629800, "6M": 15778800, "1y": 31557600}
HTF = {"1h": "4h", "4h": "1d", "1d": "1w", "5d": "2w", "1w": "1M", "2w": "1M"}
NATIVE_LIMIT = {"1h": INTRADAY_KEEP, "4h": INTRADAY_KEEP}
FULL_HISTORY = {"1d", "1w", "1M"}

DEFAULT_CFG = {"wm": 0.35, "gb": 0.4, "kh": 0.5, "km": 0.5}
GRID = {"wm": [0.0, 0.35, 0.7], "gb": [0.2, 0.4, 0.6], "kh": [0.0, 0.5, 1.0], "km": [0.0, 0.5, 1.0]}

IND_WEIGHTS = {"Lage in 100-Kerzen-Spanne": 2.5, "Abstand EMA 50": 1.5, "Abstand SMA 200": 1.5,
               "Bollinger %B": 1.0, "RSI (14)": 0.75, "Stochastic RSI": 0.5, "Williams %R": 0.5,
               "CCI (20)": 0.5, "MFI (14)": 0.75,
               "Erholung vom Tief": 2.0, "Stoch-RSI Kreuzung": 1.0, "RSI-Schwung (3)": 1.0,
               "MACD dreht": 1.0, "RSI-Divergenz": 1.0, "Volumen-Klimax": 0.5, "Kerzen-Docht": 0.5,
               "Fear & Greed": 1.0, "Dollar-Index (DXY)": 1.0, "Bitcoin/Gold": 1.0, "BTC-Dominanz": 1.0}
IND_INFO = {
    "RSI (14)": "Relative Stärke der letzten 14 Kerzen. Unter 30 = überverkauft (Kauf), über 70 = überkauft (Verkauf).",
    "Stochastic RSI": "Wo steht der RSI in seiner eigenen Spanne? Ganz unten = Kauf, ganz oben = Verkauf.",
    "Williams %R": "Schlusskurs in der 14-Kerzen-Spanne. Nahe Tief = Kauf, nahe Hoch = Verkauf.",
    "CCI (20)": "Abweichung vom Durchschnittspreis. Stark negativ = Kauf, stark positiv = Verkauf.",
    "MFI (14)": "RSI mit Volumen. Niedrig = Geld abgeflossen (Kauf), hoch = überhitzt (Verkauf).",
    "Bollinger %B": "Lage zwischen den Bollinger-Bändern. Unteres Band = Kauf, oberes = Verkauf.",
    "Abstand EMA 50": "Wie weit liegt der Kurs unter/über dem 50er-Durchschnitt (in ATR gemessen)?",
    "Abstand SMA 200": "Wie weit liegt der Kurs unter/über dem langfristigen 200er-Durchschnitt?",
    "Lage in 100-Kerzen-Spanne": "0 % = Kurs am Tief der letzten 100 Kerzen (Kauf), 100 % = am Hoch (Verkauf).",
    "MACD dreht": "Wird das MACD-Histogramm gerade größer (dreht hoch) oder kleiner (dreht runter)?",
    "RSI-Divergenz": "Kurs macht neues Tief, RSI aber nicht = bullisch. Umgekehrt am Hoch = bärisch.",
    "Stoch-RSI Kreuzung": "Kreuzt die schnelle K-Linie die langsame D-Linie nach oben (Kauf) oder unten?",
    "RSI-Schwung (3)": "Steigt oder fällt der RSI in den letzten 3 Kerzen?",
    "Erholung vom Tief": "Hat sich der Kurs schon vom Tief der letzten 6 Kerzen gelöst (Kauf) bzw. vom Hoch (Verkauf)?",
    "Volumen-Klimax": "Rote Kerze mit sehr hohem Volumen = Ausverkauf (Kauf), grüne = Kaufrausch (Verkauf).",
    "Kerzen-Docht": "Langer unterer Docht = Käufer wehren ab (Kauf), langer oberer = Verkäufer (Verkauf).",
    "Fear & Greed": "Stimmung am Kryptomarkt (alternative.me). Extreme Angst = Kauf, extreme Gier = Verkauf.",
    "Dollar-Index (DXY)": "Stärke des US-Dollars. Steigender Dollar bremst Krypto (Verkauf), fallender hilft (Kauf).",
    "Bitcoin/Gold": "Bitcoin-Preis in Gold gemessen, verglichen mit dem Durchschnitt des letzten Jahres. Billig = Kauf.",
    "BTC-Dominanz": "Nur Altcoins. Steigt Bitcoin gegenüber den Altcoins, fließt Geld ab (Verkauf), fällt sie = Kauf. "
                    "Näherung aus den Top-20-Coins.",
}
GROUP_OF = {k: ("Lage" if i < 9 else "Wende" if i < 16 else "Markt") for i, k in enumerate(IND_INFO)}

STABLE_SYMBOLS = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "USDD", "FRAX", "LUSD",
    "GUSD", "USDP", "BUSD", "EURC", "EURS", "EURT", "USD0", "USDX", "RLUSD", "USDG", "USD1",
    "USDTB", "BFUSD", "SUSDE", "SUSDS", "USDY", "USYC", "BUIDL", "USDF", "USDB", "CRVUSD",
    "GHO", "DOLA", "MIM", "USTC", "XAUT", "PAXG", "USDA", "USDO", "AUSD", "FXUSD", "USDN",
    "SDAI", "OUSG", "USDM", "USR", "DEUSD", "USDZ", "USDL", "USDQ", "FRXUSD", "AVUSD", "USDAI",
}
WRAPPED_SYMBOLS = {
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "EETH", "CBBTC", "RETH", "METH", "EZETH",
    "WBETH", "SOLVBTC", "LBTC", "BTCB", "JITOSOL", "MSOL", "BNSOL", "CLBTC", "RSETH", "OSETH",
    "CBETH", "SAVAX", "WBNB", "WTRX", "TBTC", "JSOL", "SWETH", "PUFETH", "LSETH",
    "FBTC", "UNIBTC", "PUMPBTC", "EBTC", "SFRXETH", "FRXETH", "WHYPE", "KHYPE", "STHYPE",
    "BBSOL", "HSOL", "STS", "WS", "XSOLVBTC", "ENABTC", "WEETHS", "RSWETH", "ETHX", "TETH",
}
NAME_SKIP = ("wrapped", "staked", "bridged", "restaked", "liquid staking", "tokenized", "bitcoin bep2")

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "krypto-dashboard/4.0 (+github-actions)"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class Blocked(Exception):
    pass


class NotFound(Exception):
    pass


def log(*a):
    print(*a, flush=True)


def http_get(url, params=None, tries=4, timeout=25, headers=None, as_text=False):
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After") or 0) or 15 * (i + 1)
                log(f"  Rate-Limit – warte {wait}s")
                time.sleep(min(wait, 60))
                continue
            if r.status_code in (403, 451):
                raise Blocked(f"HTTP {r.status_code}")
            if r.status_code in (400, 404):
                raise NotFound(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text if as_text else r.json()
        except (Blocked, NotFound):
            raise
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


# ----------------------------------------------------------------- Coin-Liste
def coins_from_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    rows = http_get(url, {"vs_currency": "usd", "order": "market_cap_desc", "per_page": CANDIDATES,
                          "page": 1, "sparkline": "false", "price_change_percentage": "24h"})
    stable_ids = set()
    try:
        time.sleep(3)
        st = http_get(url, {"vs_currency": "usd", "category": "stablecoins", "per_page": 250, "page": 1})
        stable_ids = {c["id"] for c in st}
    except Exception as e:
        log("  Stablecoin-Kategorie nicht abrufbar:", e)
    return [{"sym": (c.get("symbol") or "").upper(), "name": c.get("name") or "",
             "rank": c.get("market_cap_rank") or 9999, "price": c.get("current_price"),
             "mcap": c.get("market_cap"), "chg": c.get("price_change_percentage_24h"),
             "stable": c.get("id") in stable_ids} for c in rows]


def coins_from_coinpaprika():
    rows = http_get("https://api.coinpaprika.com/v1/tickers", {"quotes": "USD"}, timeout=40)
    rows = sorted([r for r in rows if r.get("rank")], key=lambda r: r["rank"])[:CANDIDATES]
    out = []
    for c in rows:
        q = (c.get("quotes") or {}).get("USD", {})
        out.append({"sym": (c.get("symbol") or "").upper(), "name": c.get("name") or "",
                    "rank": c.get("rank"), "price": q.get("price"), "mcap": q.get("market_cap"),
                    "chg": q.get("percent_change_24h"), "stable": False})
    return out


def is_excluded(c):
    sym, name, price = c["sym"], c["name"].lower(), c["price"] or 0
    if c["stable"] or sym in STABLE_SYMBOLS or sym in WRAPPED_SYMBOLS:
        return True
    if any(k in name for k in NAME_SKIP) or "stablecoin" in name:
        return True
    return ("USD" in sym or "usd" in name) and 0.95 < price < 1.05


def load_coins():
    for fn in (coins_from_coingecko, coins_from_coinpaprika):
        try:
            coins = fn()
            if coins:
                log(f"Coin-Liste über {fn.__name__}: {len(coins)} Kandidaten")
                return sorted([c for c in coins if not is_excluded(c)], key=lambda c: c["rank"])
        except Exception as e:
            log(f"{fn.__name__} fehlgeschlagen: {e}")
    raise SystemExit("Keine Coin-Liste abrufbar – Abbruch.")


# ----------------------------------------------------------------- Kerzen
def to_df(rows):
    rows = sorted({int(r[0]): r for r in rows}.values(), key=lambda r: int(r[0]))
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    return df.astype({"t": "int64", "o": float, "h": float, "l": float, "c": float, "v": float})


def binance(base):
    def fetch(sym, tf):
        if tf in FULL_HISTORY:   # komplette Historie seitenweise laden
            rows, start = [], 0
            for _ in range(12):
                data = http_get(f"{base}/api/v3/klines", {"symbol": f"{sym}USDT", "interval": tf,
                                                          "limit": 1000, "startTime": start})
                rows += data
                if len(data) < 1000:
                    break
                start = int(data[-1][0]) + 1
                time.sleep(0.1)
        else:
            rows = http_get(f"{base}/api/v3/klines",
                            {"symbol": f"{sym}USDT", "interval": tf, "limit": NATIVE_LIMIT[tf]})
        return to_df([(int(k[0]) // 1000, k[1], k[2], k[3], k[4], k[5]) for k in rows])
    return fetch


OKX_BAR = {"1h": "1Hutc", "4h": "4Hutc", "1d": "1Dutc", "1w": "1Wutc", "1M": "1Mutc"}


def okx(sym, tf):
    data = http_get("https://www.okx.com/api/v5/market/candles",
                    {"instId": f"{sym}-USDT", "bar": OKX_BAR[tf], "limit": 300})
    if str(data.get("code")) != "0" or not data.get("data"):
        raise NotFound(data.get("msg") or "leer")
    return to_df([(int(k[0]) // 1000, k[1], k[2], k[3], k[4], k[5]) for k in data["data"]])


KUCOIN_TYPE = {"1h": "1hour", "4h": "4hour", "1d": "1day", "1w": "1week"}


def kucoin(sym, tf):
    if tf == "1M":
        return aggregate(kucoin(sym, "1d"), "month")
    end = int(time.time())
    data = http_get("https://api.kucoin.com/api/v1/market/candles",
                    {"type": KUCOIN_TYPE[tf], "symbol": f"{sym}-USDT",
                     "startAt": end - 1500 * TF_SEC[tf], "endAt": end})
    if str(data.get("code")) != "200000" or not data.get("data"):
        raise NotFound(data.get("msg") or "leer")
    return to_df([(int(k[0]), k[1], k[3], k[4], k[2], k[5]) for k in data["data"]])


SOURCES = [("Binance", binance("https://data-api.binance.vision")),
           ("Binance", binance("https://api.binance.com")),
           ("OKX", okx), ("KuCoin", kucoin)]
DEAD = set()


def aggregate(df, how):
    if df is None or df.empty:
        return df
    dt = pd.to_datetime(df["t"], unit="s", utc=True)
    if how == "5d":
        key = df["t"] // (5 * 86400)
    elif how == "2w":
        key = (df["t"] - 4 * 86400) // (14 * 86400)
    elif how == "month":
        key = dt.dt.year * 12 + dt.dt.month
    elif how == "6M":
        key = dt.dt.year * 2 + (dt.dt.month - 1) // 6
    else:
        key = dt.dt.year
    g = df.groupby(np.asarray(key), sort=True)
    out = pd.DataFrame({"t": g["t"].first(), "o": g["o"].first(), "h": g["h"].max(),
                        "l": g["l"].min(), "c": g["c"].last(), "v": g["v"].sum()}).reset_index(drop=True)
    if how in ("5d", "2w") and len(out) > 3:
        out = out.iloc[1:].reset_index(drop=True)
    return out


def load_all(sym):
    for i, (name, fetch) in enumerate(SOURCES):
        if i in DEAD:
            continue
        try:
            base = fetch(sym, "4h")
            if len(base) < SCORE_MIN:
                continue
        except Blocked as e:
            log(f"  {name} (Quelle {i}) gesperrt ({e}) – übersprungen")
            DEAD.add(i)
            continue
        except Exception:
            continue
        frames = {"4h": base}
        for tf in ("1h", "1d", "1w", "1M"):
            try:
                time.sleep(0.1)
                frames[tf] = fetch(sym, tf)
            except Exception:
                pass
        if "1d" in frames:
            frames["5d"] = aggregate(frames["1d"], "5d")
        if "1w" in frames:
            frames["2w"] = aggregate(frames["1w"], "2w")
        if "1M" in frames:
            frames["6M"] = aggregate(frames["1M"], "6M")
            frames["1y"] = aggregate(frames["1M"], "1y")
        return name, {k: v for k, v in frames.items() if v is not None and len(v) >= 2}
    return None, None


# ----------------------------------------------------------------- Markt-Daten
def yahoo_daily(symbol):
    data = http_get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    {"range": "max", "interval": "1d"}, headers={"User-Agent": BROWSER_UA}, tries=2)
    res = data["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=np.asarray(res["timestamp"], dtype="int64"),
                  dtype=float).dropna()
    s.index = (s.index // 86400) * 86400
    return s[~s.index.duplicated(keep="last")]


def fred_daily(series_id):
    txt = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": series_id},
                   headers={"User-Agent": BROWSER_UA}, as_text=True, tries=2)
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    idx = (pd.to_datetime(df["date"]).astype("int64") // 10**9).astype("int64").values
    return pd.Series(df["v"].values, index=idx, dtype=float)


def load_macro():
    m = {}
    try:
        d = http_get("https://api.alternative.me/fng/", {"limit": 0, "format": "json"})["data"]
        s = pd.Series({int(x["timestamp"]): float(x["value"]) for x in d}).sort_index()
        m["fng"] = (s, 3600)  # Wert gilt ab ~1 Std nach Tagesbeginn
        m["fng_label"] = d[0].get("value_classification", "")
        log(f"Fear & Greed: {len(s)} Tage")
    except Exception as e:
        log("Fear & Greed nicht abrufbar:", e)
    for key, sources in (("dxy", [lambda: yahoo_daily("DX-Y.NYB"), lambda: fred_daily("DTWEXBGS")]),
                         ("gold", [lambda: yahoo_daily("GC=F"),
                                   lambda: binance("https://data-api.binance.vision")("PAXG", "1d").set_index("t")["c"]])):
        for fn in sources:
            try:
                s = fn().sort_index()
                if len(s) > 100:
                    m[key] = (s, 22 * 3600)  # Tagesschluss USA
                    log(f"{key}: {len(s)} Tage")
                    break
            except Exception as e:
                log(f"{key}-Quelle fehlgeschlagen: {e}")
    try:
        g = http_get("https://api.coingecko.com/api/v3/global")["data"]
        m["global"] = {"mcap": g["total_market_cap"].get("usd"),
                       "mcap_ch": g.get("market_cap_change_percentage_24h_usd"),
                       "dom": g["market_cap_percentage"].get("btc")}
    except Exception as e:
        log("CoinGecko global nicht abrufbar:", e)
    return m


# ----------------------------------------------------------------- Indikatoren
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def sma(s, n):
    return s.rolling(n).mean()


def lin(x, at100, at0):
    return (100 * (x - at0) / (at100 - at0)).clip(0, 100)


def tanh_score(x):
    return 50 + 50 * np.tanh(x)


def rolling_mad(x, n):
    a = np.asarray(x, dtype=float)
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n)
        out[n - 1:] = np.mean(np.abs(w - w.mean(axis=1, keepdims=True)), axis=1)
    return pd.Series(out)


def close_times(t):
    t = np.asarray(t, dtype="int64")
    step = int(np.median(np.diff(t))) if len(t) > 1 else 86400
    return np.r_[t[1:], t[-1] + step]


def align(ct, src_t, src_v):
    """Für jede Kerze (Schlusszeit ct) den letzten bis dahin bekannten Wert."""
    idx = np.searchsorted(src_t, ct, side="right") - 1
    out = np.where(idx >= 0, np.asarray(src_v, dtype=float)[np.clip(idx, 0, None)], np.nan)
    return out


def macro_scores(m, btc_daily):
    """Tägliche Teil-Scores der Markt-Indikatoren: {name: (verfügbar_ab, score, wert, text-funktion)}."""
    out = {}
    if "fng" in m:
        s, lag = m["fng"]
        sc = lin(s, 15, 85)
        out["Fear & Greed"] = (s.index.values + lag, sc.values, s.values,
                               lambda v: ("extreme Angst" if v < 25 else "Angst" if v < 45 else "neutral" if v <= 55
                                          else "Gier" if v <= 75 else "extreme Gier"), lambda v: f"{v:.0f}")
    if "dxy" in m:
        s, lag = m["dxy"]
        z = (s / ema(s, 50) - 1) / 0.015
        out["Dollar-Index (DXY)"] = (s.index.values + lag, (50 - 50 * np.tanh(z / 1.5)).values, z.values,
                                     lambda v: "Dollar stark (Gegenwind)" if v > 0.5 else "Dollar schwach (Rückenwind)"
                                     if v < -0.5 else "neutral", lambda v: f"{v:+.1f}")
    if "gold" in m and btc_daily is not None and len(btc_daily) > 400:
        g, lag = m["gold"]
        bt = pd.Series(btc_daily["c"].values, index=(btc_daily["t"].values // 86400) * 86400)
        gd = g.reindex(bt.index, method="ffill")
        lr = np.log(bt / gd)
        z = (lr - lr.rolling(365, min_periods=200).mean()) / lr.rolling(365, min_periods=200).std()
        out["Bitcoin/Gold"] = (bt.index.values + 86400, (50 - 50 * np.tanh(z / 1.5)).values, z.values,
                               lambda v: "BTC günstig ggü. Gold" if v < -0.75 else "BTC teuer ggü. Gold" if v > 0.75
                               else "normal", lambda v: f"{v:+.1f} σ")
    return out


def components(df, macro, dom=None):
    """Alle Teil-Scores einer Kerzenreihe + Gruppen (Lage, Wende, Markt)."""
    o, h, l, c, v = df["o"], df["h"], df["l"], df["c"], df["v"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14).replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    delta = c.diff()
    rs = rma(delta.clip(lower=0), 14) / rma((-delta).clip(lower=0), 14).replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(100).where(rma(delta.abs(), 14).notna())
    rmin, rmax = rsi.rolling(14).min(), rsi.rolling(14).max()
    stoch = (((rsi - rmin) / (rmax - rmin).replace(0, np.nan)).fillna(0.5) * 100).where(rmin.notna())
    k = sma(stoch, 3)
    d = sma(k, 3)
    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    wr = -100 * (hh14 - c) / (hh14 - ll14).replace(0, np.nan)
    tp = (h + l + c) / 3
    cci = (tp - sma(tp, 20)) / (0.015 * rolling_mad(tp, 20).replace(0, np.nan))
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    mfi = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).fillna(100).where(pos.notna())
    bb_std = c.rolling(20).std(ddof=0)
    pctb = (c - (sma(c, 20) - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    z50 = (c - ema(c, 50)) / atr
    z200 = (c - sma(c, 200)) / atr
    hh100, ll100 = h.rolling(100, min_periods=50).max(), l.rolling(100, min_periods=50).min()
    rpos = (c - ll100) / (hh100 - ll100).replace(0, np.nan)
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    hist_turn = (hist - hist.shift(2)) / (0.15 * atr)
    n = 14
    lo_now, lo_prev = l.rolling(n).min(), l.shift(n).rolling(n).min()
    hi_now, hi_prev = h.rolling(n).max(), h.shift(n).rolling(n).max()
    r_lo_now, r_lo_prev = rsi.rolling(n).min(), rsi.shift(n).rolling(n).min()
    r_hi_now, r_hi_prev = rsi.rolling(n).max(), rsi.shift(n).rolling(n).max()
    bull = ((lo_now < lo_prev) & (r_lo_now > r_lo_prev)) * ((r_lo_now - r_lo_prev) / 8).clip(0, 1)
    bear = ((hi_now > hi_prev) & (r_hi_now < r_hi_prev)) * ((r_hi_prev - r_hi_now) / 8).clip(0, 1)
    div = (50 + 45 * bull.fillna(0) - 45 * bear.fillna(0)).where(r_lo_prev.notna())
    rsi_mom = rsi - rsi.shift(3)
    vr = v / sma(v, 20).replace(0, np.nan)
    body = ((c - o) / rng).fillna(0)
    climax = 50 - 50 * np.tanh(1.5 * (vr - 1.3).clip(lower=0)) * body
    wick = (((np.minimum(o, c) - l) - (h - np.maximum(o, c))) / rng).fillna(0).rolling(2).mean()
    bounce = ((c - l.rolling(6).min()) - (h.rolling(6).max() - c)) / atr

    f1 = lambda x: f"{x:.1f}"
    # Name -> (Teil-Score, Rohwert, Wert-Format, Erklärung)
    ind = {
        "RSI (14)": (lin(rsi, 25, 75), rsi, f1,
                     lambda x: "überverkauft" if x < 30 else "überkauft" if x > 70 else "neutral"),
        "Stochastic RSI": (lin(k, 10, 90), k, f1, lambda x: "ganz unten" if x < 15 else "ganz oben" if x > 85 else "Mitte"),
        "Williams %R": (lin(wr, -90, -10), wr, f1,
                        lambda x: "überverkauft" if x < -80 else "überkauft" if x > -20 else "neutral"),
        "CCI (20)": (lin(cci, -200, 200), cci, lambda x: f"{x:.0f}",
                     lambda x: "weit unter Schnitt" if x < -100 else "weit über Schnitt" if x > 100 else "normal"),
        "MFI (14)": (lin(mfi, 15, 85), mfi, f1,
                     lambda x: "Geld fließt stark ab" if x < 20 else "überhitzt" if x > 80 else "neutral"),
        "Bollinger %B": (lin(pctb, 0, 1), pctb, lambda x: f"{x:.2f}",
                         lambda x: "unteres Band" if x < 0.1 else "oberes Band" if x > 0.9 else "innerhalb"),
        "Abstand EMA 50": (lin(z50, -5, 5), z50, lambda x: f"{x:.1f} ATR",
                           lambda x: "weit darunter" if x < -2.5 else "weit darüber" if x > 2.5 else "nahe dran"),
        "Abstand SMA 200": (lin(z200, -12, 12), z200, lambda x: f"{x:.1f} ATR",
                            lambda x: "deutlich darunter" if x < -5 else "deutlich darüber" if x > 5 else "nahe dran"),
        "Lage in 100-Kerzen-Spanne": (lin(rpos, 0, 1), rpos, lambda x: f"{x * 100:.0f} %",
                                      lambda x: "nahe Tief" if x < 0.15 else "nahe Hoch" if x > 0.85 else "Mitte"),
        "MACD dreht": (tanh_score(hist_turn), hist_turn, lambda x: f"{x:+.2f}",
                       lambda x: "dreht nach oben" if x > 0.3 else "dreht nach unten" if x < -0.3 else "flach"),
        "RSI-Divergenz": (div, div, lambda x: f"{x:.0f}",
                          lambda x: "bullische Divergenz" if x > 60 else "bärische Divergenz" if x < 40 else "keine"),
        "Stoch-RSI Kreuzung": (tanh_score((k - d) / 10), k - d, f1,
                               lambda x: "K kreuzt hoch" if x > 2 else "K kreuzt runter" if x < -2 else "neutral"),
        "RSI-Schwung (3)": (tanh_score(rsi_mom / 8), rsi_mom, f1,
                            lambda x: "RSI steigt" if x > 2 else "RSI fällt" if x < -2 else "seitwärts"),
        "Erholung vom Tief": (tanh_score(bounce / 2), bounce, lambda x: f"{x:.1f} ATR",
                              lambda x: "hebt vom Tief ab" if x > 1 else "rutscht vom Hoch ab" if x < -1 else "unentschieden"),
        "Volumen-Klimax": (climax, vr * np.sign(body), lambda x: f"{abs(x):.1f}×",
                           lambda x: "Ausverkauf mit Volumen" if x < -1.5 else "Kaufrausch mit Volumen" if x > 1.5 else "normal"),
        "Kerzen-Docht": ((50 + 50 * wick).clip(0, 100), wick, lambda x: f"{x:.2f}",
                         lambda x: "Abweisung unten" if x > 0.25 else "Abweisung oben" if x < -0.25 else "neutral"),
    }
    ct = close_times(df["t"].values)
    for name, (avail, sc, raw, note, vf) in macro.items():
        ind[name] = (pd.Series(align(ct, avail, sc)), pd.Series(align(ct, avail, raw)), vf, note)
    if dom is not None:
        z = dom
        ind["BTC-Dominanz"] = (pd.Series(50 - 50 * np.tanh(z / 1.5)), pd.Series(z), lambda x: f"{x:+.1f} σ",
                               lambda x: "steigt (Geld fließt zu BTC)" if x > 0.5 else "fällt (gut für Altcoins)"
                               if x < -0.5 else "stabil")

    subs = pd.DataFrame({name: np.asarray(v[0], dtype=float) for name, v in ind.items()})
    groups = {}
    for g in ("Lage", "Wende", "Markt"):
        cols = [x for x in subs.columns if GROUP_OF.get(x) == g]
        if not cols:
            groups[g] = np.full(len(df), np.nan)
            continue
        w = np.array([IND_WEIGHTS.get(x, 1.0) for x in cols])
        part = subs[cols].values
        mask = ~np.isnan(part)
        with np.errstate(invalid="ignore", divide="ignore"):
            groups[g] = np.where(mask.any(axis=1), np.nansum(part * w, axis=1) / (mask * w).sum(axis=1), np.nan)
    return {"lage": groups["Lage"], "wende": groups["Wende"], "markt": groups["Markt"],
            "subs": subs, "ind": ind, "ct": ct, "t": df["t"].values, "c": df["c"].values}


def combine(C, cfg, htf_base=None, mkt_base=None):
    """Score-Formel. Gibt (Basis ohne Filter, Rohwert) zurück, beide in -1..1 (ungefähr)."""
    lage, wende, markt = C["lage"], C["wende"], C["markt"]
    wm = cfg["wm"]
    m = np.where(np.isnan(markt), lage, markt)
    with np.errstate(invalid="ignore"):
        d = ((lage - 50) + wm * (m - 50)) / (1 + wm) / 50
        turn = np.nan_to_num((wende - 50) / 50)
        gate = np.clip(cfg["gb"] + (1 - cfg["gb"]) * turn * np.sign(d), 0.05, 1.0)
        base = d * gate
        f = np.ones(len(base))
        for kk, b in ((cfg["kh"], htf_base), (cfg["km"], mkt_base)):
            if b is not None and kk > 0:
                a = np.nan_to_num(np.tanh(3 * b)) * np.sign(base)
                f = f * np.clip(1 + 0.6 * kk * a, 0.3, 1.4)
    raw = base * f
    base = base.copy()
    base[:30] = np.nan
    raw[:30] = np.nan
    return base, raw


def to_score(raw, S):
    return np.clip(50 + 50 * raw * S, 0, 100)


def htf_map(C_low, C_high):
    """Index der letzten abgeschlossenen übergeordneten Kerze für jede Kerze."""
    return np.searchsorted(C_high["ct"], C_low["ct"], side="right") - 1


def take(arr, idx):
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = arr[idx[ok]]
    return out


def pivots(df, n):
    w = 2 * n + 1
    hi = (df["h"] == df["h"].rolling(w, center=True).max()).fillna(False).values
    lo = (df["l"] == df["l"].rolling(w, center=True).min()).fillna(False).values
    return hi, lo


# ----------------------------------------------------------------- Kalibrierung
def entries(score):
    s = np.nan_to_num(score, nan=50.0)
    prev = np.r_[50.0, s[:-1]]
    buy = (s >= BUY_LEVEL) & (prev < BUY_LEVEL)
    sell = (s <= SELL_LEVEL) & (prev > SELL_LEVEL)
    return buy, sell


def evaluate(items, cfg, S=None):
    """items: Liste von Kalibrier-Datensätzen. Gibt Kennzahlen für Training/Test zurück."""
    cache = {}
    raws = []
    for it in items:
        hb = None
        if it["H"] is not None:
            key = id(it["H"])
            if key not in cache:
                cache[key] = combine(it["H"], cfg)[0]
            hb = take(cache[key], it["hidx"])
        mb = None
        if it["M"] is not None:
            key = id(it["M"])
            if key not in cache:
                cache[key] = combine(it["M"], cfg)[0]
            mb = take(cache[key], it["midx"])
        raws.append(combine(it["C"], cfg, hb, mb)[1])
    if S is None:
        pool = np.concatenate([np.abs(r[it["train"]]) for r, it in zip(raws, items)])
        pool = pool[~np.isnan(pool)]
        q = np.quantile(pool, 1 - ZONE_SHARE) if len(pool) > 100 else 0.3
        S = float(np.clip(0.6 / max(q, 1e-6), 0.8, 5.0))
    res = {}
    for part in ("train", "test"):
        signed = []
        for r, it in zip(raws, items):
            sc = to_score(r, S)
            buy, sell = entries(sc)
            c = it["C"]["c"]
            H = STAT_HORIZON
            valid = np.zeros(len(c), bool)
            valid[:-H] = True
            mask = it[part] & valid
            fwd = np.full(len(c), np.nan)
            fwd[:-H] = c[H:] / c[:-H] - 1
            scale = it["scale"]
            signed += list(fwd[buy & mask] / scale) + list(-fwd[sell & mask] / scale)
        signed = np.array(signed)
        res[part] = {"n": int(len(signed)), "obj": float(np.mean(signed)) if len(signed) else -9,
                     "hit": float(np.mean(signed > 0) * 100) if len(signed) else 0}
    return S, res


def calibrate(items):
    t0 = time.time()
    S0, base_res = evaluate(items, DEFAULT_CFG)
    best = (DEFAULT_CFG, S0, base_res)
    for wm in GRID["wm"]:
        for gb in GRID["gb"]:
            for kh in GRID["kh"]:
                for km in GRID["km"]:
                    cfg = {"wm": wm, "gb": gb, "kh": kh, "km": km}
                    S, r = evaluate(items, cfg)
                    if r["train"]["n"] >= 40 and r["train"]["obj"] > best[2]["train"]["obj"]:
                        best = (cfg, S, r)
    cfg, S, r = best
    use_best = r["test"]["n"] >= 20 and r["test"]["obj"] >= base_res["test"]["obj"]
    chosen = (cfg, S, r) if use_best else (DEFAULT_CFG, S0, base_res)
    log(f"Kalibrierung ({time.time() - t0:.0f}s): bester Train {r['train']}, Test {r['test']}; "
        f"Standard Test {base_res['test']}; übernommen: {'neu' if use_best else 'Standard'}")
    return {"cfg": chosen[0], "S": chosen[1], "res": chosen[2], "default": base_res,
            "adopted": bool(use_best), "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def previous_calibration():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    url = (f"https://{owner.lower()}.github.io/calib.json" if name.lower() == f"{owner.lower()}.github.io"
           else f"https://{owner.lower()}.github.io/{name}/calib.json")
    try:
        c = http_get(url, tries=1)
        age = time.time() - datetime.strptime(c["date"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        if age < CALIB_MAX_AGE_DAYS * 86400 and "cfg" in c and "S" in c:
            log(f"Kalibrierung von {c['date']} wird weiterverwendet.")
            return c
    except Exception as e:
        log("Keine frühere Kalibrierung gefunden:", e)
    return None


# ----------------------------------------------------------------- Gesamtmarkt-Index
def build_index(dfs, weights):
    """Gewichteter Index aus mehreren Coins (Log-Renditen), als OHLCV-Tabelle."""
    if not dfs:
        return None
    parts = []
    for df, w in zip(dfs, weights):
        pc = df["c"].shift()
        parts.append(pd.DataFrame({"t": df["t"], "w": w, "rc": np.log(df["c"] / pc), "rh": np.log(df["h"] / pc),
                                   "rl": np.log(df["l"] / pc), "qv": df["v"] * df["c"]}))
    a = pd.concat(parts).dropna()
    if a.empty:
        return None
    g = a.assign(wc=a.w * a.rc, wh=a.w * a.rh, wl=a.w * a.rl).groupby("t").agg(
        w=("w", "sum"), wc=("wc", "sum"), wh=("wh", "sum"), wl=("wl", "sum"), qv=("qv", "sum"))
    rc, rh, rl = g.wc / g.w, g.wh / g.w, g.wl / g.w
    c = 100 * np.exp(rc.cumsum())
    o = c.shift().fillna(100)
    h = np.maximum(o * np.exp(rh), np.maximum(o, c))
    l = np.minimum(o * np.exp(rl), np.minimum(o, c))
    return pd.DataFrame({"t": g.index.values.astype("int64"), "o": o.values, "h": h.values, "l": l.values,
                         "c": c.values, "v": g.qv.values})


def dominance_z(btc_df, alt_idx, t):
    """BTC gegenüber Altcoin-Index: Abweichung vom 50er-Schnitt in Standardabweichungen."""
    if btc_df is None or alt_idx is None:
        return None
    s = pd.Series(np.log(btc_df["c"].values), index=btc_df["t"].values)
    a = pd.Series(np.log(alt_idx["c"].values), index=alt_idx["t"].values)
    r = (s - a).dropna()
    if len(r) < 60:
        return None
    z = (r - ema(r, 50)) / r.rolling(100, min_periods=40).std()
    return z.reindex(t).values


# ----------------------------------------------------------------- Ausgabe
def sig(x, digits=6):
    if x is None or not np.isfinite(x) or x == 0:
        return 0
    return float(f"{x:.{digits}g}")


def signal_stats(closes, score):
    res = {}
    buy, sell = entries(score)
    H = STAT_HORIZON
    for kind, m in (("buy", buy), ("sell", sell)):
        idx = np.where(m[:-H])[0] if len(m) > H else []
        rets = [closes[i + H] / closes[i] - 1 for i in idx]
        if rets:
            hits = sum(1 for r in rets if (r > 0 if kind == "buy" else r < 0))
            res[kind] = [len(rets), round(100 * float(np.mean(rets)), 2), round(100 * hits / len(rets))]
    return res


def tf_output(df, C, score, htf_info, mkt_score):
    out = {}
    piv_n = PIVOT_N if len(df) >= 60 else 3
    hi, lo = pivots(df, piv_n)
    out["c"] = [[int(r.t), sig(r.o), sig(r.h), sig(r.l), sig(r.c)] for r in df.itertuples()]
    out["p"] = [[int(df["t"].iloc[i]), "H" if hi[i] else "T"] for i in range(len(df)) if hi[i] or lo[i]]
    if score is not None and np.isfinite(score).any():
        out["z"] = [None if np.isnan(x) else round(float(x), 1) for x in score]
        out["stat"] = signal_stats(df["c"].values, score)
        last = len(df) - 1
        det = []
        for name, (sc, raw, vf, note) in C["ind"].items():
            sv = float(np.asarray(sc, dtype=float)[last])
            rv = float(np.asarray(raw, dtype=float)[last])
            if np.isnan(sv):
                continue
            det.append([name, GROUP_OF.get(name, "Markt"), round(sv),
                        "–" if np.isnan(rv) else vf(rv), "–" if np.isnan(rv) else note(rv)])
        out["b"] = det
        out["k"] = {g: (None if np.isnan(C[key][last]) else round(float(C[key][last])))
                    for g, key in (("Lage", "lage"), ("Wende", "wende"), ("Markt", "markt"))}
        out["f"] = {"htf": htf_info, "mkt": None if mkt_score is None or np.isnan(mkt_score) else round(float(mkt_score))}
    return out


def last_valid(arr):
    if arr is None:
        return None
    for x in reversed(arr):
        if x is not None:
            return x
    return None


def safe_name(sym):
    return re.sub(r"[^A-Za-z0-9]", "_", sym)


def run(coins_loader=load_coins, macro_loader=load_macro, html=None):
    started = time.time()
    coins = coins_loader()
    macro = macro_loader()
    os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)

    # 1) Kerzen laden
    loaded = []
    seen = set()
    for meta in coins:
        if len(loaded) >= TOP_N:
            break
        sym = meta["sym"]
        if not sym or sym in seen:
            continue
        src, frames = load_all(sym)
        if not frames:
            log(f"  {sym}: keine Kerzen – übersprungen")
            continue
        seen.add(sym)
        loaded.append((meta, src, frames))
        log(f"  {len(loaded):3d}. {sym:<8} {src} · " + ", ".join(f"{k}:{len(v)}" for k, v in frames.items()))
    if not loaded:
        raise SystemExit("Keine Daten – Abbruch.")
    log(f"Kerzen geladen nach {time.time() - started:.0f}s")

    # 2) Markt-Index, Dominanz, Markt-Indikatoren
    btc = next((f for m, s, f in loaded if m["sym"] == "BTC"), None)
    mscores = macro_scores(macro, btc.get("1d") if btc else None)
    top = loaded[:INDEX_COINS]
    idx_df, alt_df = {}, {}
    for tf, _ in TIMEFRAMES:
        dfs = [(f[tf], m["mcap"] or 1) for m, s, f in top if tf in f]
        idx_df[tf] = build_index([d for d, _ in dfs], [w for _, w in dfs])
        alts = [(f[tf], m["mcap"] or 1) for m, s, f in top if tf in f and m["sym"] != "BTC"]
        alt_df[tf] = build_index([d for d, _ in alts], [w for _, w in alts])
    idx_C = {tf: components(d, mscores) for tf, d in idx_df.items() if d is not None and len(d) >= SCORE_MIN}

    # 3) Teil-Scores aller Coins
    comps = []
    for meta, src, frames in loaded:
        cc = {}
        for tf, df in frames.items():
            if len(df) < SCORE_MIN:
                continue
            dom = None
            if meta["sym"] != "BTC" and btc is not None and tf in btc:
                dom = dominance_z(btc[tf], alt_df.get(tf), df["t"].values)
            cc[tf] = components(df, mscores, dom)
        comps.append(cc)
    log(f"Indikatoren berechnet nach {time.time() - started:.0f}s")

    # 4) Kalibrierung
    calib = previous_calibration()
    if calib is None:
        items = []
        for (meta, src, frames), cc in list(zip(loaded, comps))[:CALIB_COINS]:
            for tf in ("4h", "1d"):
                if tf not in cc:
                    continue
                C = cc[tf]
                H = cc.get(HTF[tf])
                M = idx_C.get(tf)
                n = len(C["c"])
                cut = int(n * 0.7)
                train = np.zeros(n, bool)
                train[:cut] = True
                fwd = np.abs(C["c"][STAT_HORIZON:] / C["c"][:-STAT_HORIZON] - 1)
                items.append({"C": C, "H": H, "hidx": htf_map(C, H) if H else None,
                              "M": M, "midx": exact_index(M["t"], C["t"]) if M else None,
                              "train": train, "test": ~train, "scale": float(np.nanmedian(fwd)) or 0.01})
        if items:
            calib = calibrate(items)
        else:
            calib = {"cfg": DEFAULT_CFG, "S": 2.0, "res": None, "default": None, "adopted": False,
                     "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    cfg, S = calib["cfg"], calib["S"]

    # 5) Endgültige Scores + Ausgabe
    idx_base = {tf: combine(C, cfg)[0] for tf, C in idx_C.items()}
    listing = []
    for (meta, src, frames), cc in zip(loaded, comps):
        bases = {tf: combine(C, cfg)[0] for tf, C in cc.items()}
        tfs = {}
        for tf, _ in TIMEFRAMES:
            df = frames.get(tf)
            if df is None:
                continue
            C = cc.get(tf)
            score = htf_info = mkt_last = None
            if C is not None:
                hb = mb = None
                htf = HTF.get(tf)
                if htf in cc:
                    hidx = htf_map(C, cc[htf])
                    hb = take(bases[htf], hidx)
                    hscore = to_score(hb, S)
                    hv = hscore[-1]
                    htf_info = [htf, None if np.isnan(hv) else round(float(hv))]
                if tf in idx_C:
                    mb = take(idx_base[tf], exact_index(idx_C[tf]["t"], C["t"]))
                    mkt_last = to_score(mb, S)[-1]
                raw = combine(C, cfg, hb, mb)[1]
                score = to_score(raw, S)
            tfs[tf] = tf_output(df, C or {}, score, htf_info, mkt_last)
        fname = safe_name(meta["sym"])
        with open(os.path.join(OUT_DIR, "data", fname + ".json"), "w", encoding="utf-8") as f:
            json.dump({"s": meta["sym"], "src": src, "tf": tfs}, f, separators=(",", ":"))
        listing.append({"s": meta["sym"], "n": meta["name"], "r": meta["rank"], "f": fname, "src": src,
                        "p": sig(meta["price"] or frames["4h"]["c"].iloc[-1]),
                        "ch": None if meta["chg"] is None else round(float(meta["chg"]), 2),
                        "sc": {tf: last_valid(tfs.get(tf, {}).get("z")) for tf in LIST_TFS}})
    listing.sort(key=lambda c: c["r"])

    mk = {}
    if "fng" in macro:
        mk["fng"] = [float(macro["fng"][0].iloc[-1]), macro.get("fng_label", "")]
    if "dxy" in macro:
        mk["dxy"] = round(float(macro["dxy"][0].iloc[-1]), 2)
    if "global" in macro:
        mk.update(macro["global"])
    payload = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "buy": BUY_LEVEL, "sell": SELL_LEVEL, "h": STAT_HORIZON, "tfs": TIMEFRAMES,
               "ltfs": LIST_TFS, "iw": IND_WEIGHTS, "info": IND_INFO, "go": GROUP_OF, "htf": HTF,
               "cal": {k: calib.get(k) for k in ("cfg", "S", "res", "default", "adopted", "date")},
               "mkt": mk, "coins": listing}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=float).replace("</", "<\\/")
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html.replace("__DATA__", data))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"updated": payload["updated"]}, f)
    with open(os.path.join(OUT_DIR, "calib.json"), "w") as f:
        json.dump(calib, f, default=float)
    log(f"Fertig: {len(listing)} Coins in {time.time() - started:.0f}s")


def exact_index(src_t, t):
    """Position gleicher Zeitstempel (oder -1)."""
    pos = np.searchsorted(src_t, t)
    pos_c = np.clip(pos, 0, len(src_t) - 1)
    return np.where(np.asarray(src_t)[pos_c] == t, pos_c, -1)


HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0e14">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Krypto Score Dashboard</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='22' fill='%230b0e14'/><path d='M18 70 L40 48 L55 60 L82 30' stroke='%2322c55e' stroke-width='10' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>">
<style>
:root{--bg:#0b0e14;--card:#131821;--card2:#1a202c;--line:#242b38;--txt:#e6e9ef;--mut:#8b93a5;
--buy:#22c55e;--sell:#ef4444;--acc:#60a5fa}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif}
button{font-family:inherit}
header{position:sticky;top:0;z-index:5;background:rgba(11,14,20,.94);backdrop-filter:blur(10px);
padding:calc(10px + env(safe-area-inset-top)) 14px 10px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0;font-weight:700}
.hrow{display:flex;align-items:center;gap:10px;justify-content:space-between}
.fresh{border:1px solid var(--line);background:var(--card);color:var(--txt);border-radius:999px;padding:7px 12px;
font-size:12.5px;font-weight:600;display:flex;align-items:center;gap:7px;white-space:nowrap}
.fresh i{width:8px;height:8px;border-radius:50%;display:inline-block}
.fresh span{font-size:15px;line-height:1}
.fresh.spin span{display:inline-block;animation:sp 1s linear infinite}@keyframes sp{to{transform:rotate(360deg)}}
.mbar{display:flex;gap:6px;margin-top:8px;overflow-x:auto;scrollbar-width:none}
.mbar::-webkit-scrollbar{display:none}
.mb{flex:none;background:var(--card);border-radius:10px;padding:4px 9px;font-size:11.5px;color:var(--mut)}
.mb b{color:var(--txt);font-weight:700}
.search{margin-top:9px;width:100%;padding:10px 14px;border-radius:12px;border:1px solid var(--line);
background:var(--card);color:var(--txt);font-size:16px;outline:none}
.search:focus{border-color:var(--acc)}
.chips{display:flex;gap:7px;margin-top:8px;overflow-x:auto;scrollbar-width:none;align-items:center}
.chips::-webkit-scrollbar{display:none}
.lbl{flex:none;font-size:12px;color:var(--mut);margin-right:2px}
.chip{flex:none;padding:7px 12px;border-radius:999px;border:1px solid var(--line);background:var(--card);
color:var(--txt);font-size:13px;font-weight:600}
.chip.on{background:var(--txt);color:var(--bg);border-color:var(--txt)}
.chip.b.on{background:var(--buy);border-color:var(--buy);color:#04130a}
.chip.s.on{background:var(--sell);border-color:var(--sell);color:#fff}
.chip:disabled{opacity:.35}
.chip.sm{padding:4px 10px;font-size:12px}
.stats{display:flex;gap:8px;padding:12px 14px 4px}
.stat{flex:1;background:var(--card);border-radius:12px;padding:8px 10px}
.stat b{display:block;font-size:18px}.stat span{font-size:11px;color:var(--mut)}
ul{list-style:none;margin:0;padding:6px 10px 20px}
li.row{display:grid;grid-template-columns:30px 1fr auto;gap:10px;align-items:center;background:var(--card);
border-radius:14px;padding:11px 12px;margin:6px 0;cursor:pointer}
li.row:active{background:var(--card2)}
.rk{color:var(--mut);font-size:12px;text-align:center}
.nm{min-width:0}.nm b{font-size:15px}.nm small{color:var(--mut);margin-left:6px;font-size:12px}
.px{font-size:12px;color:var(--mut);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar{height:6px;border-radius:3px;background:var(--line);margin-top:7px;position:relative;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.sc{min-width:52px;text-align:center;font-weight:800;font-size:17px;padding:6px 0;border-radius:10px}
.up{color:var(--buy)}.dn{color:var(--sell)}
.empty{color:var(--mut);text-align:center;padding:40px 0}
footer{color:var(--mut);font-size:11px;text-align:center;padding:0 20px 30px}
#detail{position:fixed;inset:0;z-index:20;background:var(--bg);overflow-y:auto;display:none;
padding-bottom:calc(30px + env(safe-area-inset-bottom))}
#detail.open{display:block}
.dh{position:sticky;top:0;z-index:6;background:rgba(11,14,20,.96);backdrop-filter:blur(10px);
padding:calc(10px + env(safe-area-inset-top)) 12px 8px;border-bottom:1px solid var(--line)}
.dh .top{display:flex;align-items:center;gap:8px}
.back{border:0;background:var(--card);color:var(--txt);width:38px;height:38px;border-radius:12px;font-size:22px;flex:none}
.dh .t{flex:1;min-width:0}.dh .t b{font-size:16px}.dh .t div{font-size:12px;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.box{margin:10px;background:var(--card);border-radius:16px;padding:10px 6px 6px;position:relative}
.box h3{font-size:13px;margin:2px 8px 6px;color:var(--mut);font-weight:600;display:flex;justify-content:space-between;align-items:center;gap:8px}
.mres{margin:2px 8px 6px;font-size:13px;height:20px;line-height:20px;color:var(--mut);white-space:nowrap;overflow:hidden;
text-overflow:ellipsis;font-variant-numeric:tabular-nums}
.mres b{font-size:14px}
.cw{position:relative;touch-action:none;user-select:none;-webkit-user-select:none}
.cwrap{position:relative}
#pc{height:330px}#zc{height:194px}
.axt{position:absolute;left:0;right:0;bottom:0;height:44px;z-index:5;touch-action:none;
background:linear-gradient(transparent,rgba(96,165,250,.07));border-top:1px solid rgba(96,165,250,.25)}
.axt span{position:absolute;right:8px;bottom:2px;font-size:9.5px;color:#60a5fa99;pointer-events:none}
.layer{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.touch{position:absolute;inset:0;z-index:4;touch-action:none}
.vl{position:absolute;top:0;bottom:0;width:0;border-left:1px dashed}
.rect{position:absolute;border:1px solid}
.dot{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;border:2px solid #fff}
.bub{font-variant-numeric:tabular-nums;position:absolute;transform:translate(-50%,-100%);background:#0b0e14ee;border:1px solid var(--line);border-radius:10px;
padding:5px 9px;font-size:12px;white-space:nowrap;line-height:1.35;text-align:center;box-shadow:0 4px 14px #0008}
.bub b{font-size:14px}
.ctrl{display:flex;align-items:center;gap:10px;margin:10px 8px 4px;font-size:12px;color:var(--mut)}
.ctrl input[type=range]{flex:1;height:34px;accent-color:var(--acc)}
.ctrl .zl{min-width:86px;text-align:right}
.nav{position:relative;height:52px;margin:4px 8px 6px;border-radius:10px;background:var(--card2);touch-action:none;overflow:hidden;cursor:grab}
.nav canvas{position:absolute;inset:0;width:100%;height:100%}
.navw{position:absolute;top:0;bottom:0;background:rgba(96,165,250,.18);border-left:2px solid var(--acc);border-right:2px solid var(--acc)}
.navh{position:absolute;left:0;right:0;bottom:3px;text-align:center;font-size:10.5px;color:var(--mut);pointer-events:none}
.now{border:1px solid var(--line);background:var(--card2);color:var(--txt);border-radius:10px;padding:6px 10px;font-size:12px;font-weight:600}
.msg{color:var(--mut);font-size:13px;padding:6px 4px;text-align:center}
.sec{margin:10px;background:var(--card);border-radius:16px;padding:12px}
.sec h4{margin:0 0 8px;font-size:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.cat{background:var(--card2);border-radius:10px;padding:8px 10px;font-size:12px;color:var(--mut)}
.cat b{display:block;font-size:17px;color:var(--txt)}
.it{font-size:12px;background:var(--card2);border-radius:10px;padding:7px 9px;margin-bottom:6px}
.it b{font-size:13px;display:flex;justify-content:space-between;gap:6px}.it span{color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td{padding:7px 4px;border-top:1px solid var(--line);vertical-align:middle}
td.v{color:var(--mut);text-align:right;white-space:nowrap}
.mini{width:50px;height:6px;border-radius:3px;background:var(--line);position:relative;display:inline-block}
.mini i{position:absolute;top:0;bottom:0;left:0;border-radius:3px}
details summary{cursor:pointer;font-weight:600;font-size:14px}
.exp p{font-size:12.5px;color:var(--mut);margin:6px 0}.exp b{color:var(--txt)}
.exp li{font-size:12px;color:var(--mut);margin:4px 0}.exp ul{padding:0 0 0 16px;list-style:disc}
.note{font-size:11px;color:var(--mut);margin:8px 12px 0;line-height:1.5}
</style>
</head>
<body>
<header>
  <div class="hrow"><h1>Krypto Score</h1><button class="fresh" id="ref"></button></div>
  <div class="mbar" id="mbar"></div>
  <input class="search" id="q" type="search" placeholder="Coin suchen (z. B. BTC, Solana)…" autocomplete="off">
  <div class="chips" id="sort"><span class="lbl">Sortieren:</span>
    <button class="chip" data-o="mc">Marktkap.</button>
    <button class="chip" data-o="hi">Score hoch → tief</button>
    <button class="chip" data-o="lo">Score tief → hoch</button>
  </div>
  <div class="chips" id="ltf"><span class="lbl">Score:</span></div>
  <div class="chips" id="chips">
    <button class="chip on" data-f="all">Alle</button>
    <button class="chip b" data-f="buy">Starker Kauf ≥80</button>
    <button class="chip" data-f="pos">Kauf 60–80</button>
    <button class="chip" data-f="neu">Neutral</button>
    <button class="chip" data-f="neg">Verkauf 20–40</button>
    <button class="chip s" data-f="sell">Starker Verkauf ≤20</button>
  </div>
</header>
<div class="stats" id="stats"></div>
<ul id="list"></ul>
<div class="sec exp" style="margin:0 10px 14px"><details><summary>Wie entsteht der Score?</summary><div class="expc"></div></details></div>
<footer>Keine Anlageberatung.</footer>

<div id="detail">
  <div class="dh">
    <div class="top">
      <button class="back" id="back" aria-label="Zurück">‹</button>
      <div class="t"><b id="dt"></b><div id="ds"></div></div>
      <button class="fresh" id="ref2"></button>
      <div class="sc" id="dsc"></div>
    </div>
    <div class="chips" id="tfc"></div>
  </div>
  <div class="box"><h3><span>Chart wischen = messen (1–2 Finger)</span>
    <button class="chip sm" id="now">Jetzt ⇥</button></h3>
    <div class="mres" id="mres"></div>
    <div class="cwrap" id="cwrap">
      <div class="cw" id="pcw"><div id="pc"></div><div class="layer" id="ml"></div><div class="touch" id="ptouch"></div></div>
      <div class="cw" id="zcw"><div id="zc"></div><div class="touch" id="ztouch"></div></div>
      <div class="layer" id="pvl"></div>
      <div class="axt" id="axt"><span>Zeitachse: ziehen = schieben · 2 Finger = zoomen</span></div>
    </div>
    <div class="msg" id="noz" style="display:none">Für diesen Zeitrahmen gibt es zu wenige Kerzen für einen Score.</div>
  </div>
  <div class="sec" id="fbox"><h4>Filter</h4><div class="grid2" id="filt"></div></div>
  <div class="sec" id="statbox"><h4>Bilanz der Signale (gesamte Historie)</h4><div class="grid2" id="stat"></div></div>
  <div class="sec" id="catbox"><h4>Bestandteile</h4><div class="grid3" id="cats"></div></div>
  <div class="sec" id="pcbox"><div class="grid2">
    <div><h4 class="up">Spricht für Kaufen</h4><div id="pro"></div></div>
    <div><h4 class="dn">Spricht für Verkaufen</h4><div id="con"></div></div>
  </div></div>
  <div class="sec" id="allbox"><details><summary>Alle Indikatoren</summary><table id="all"></table></details></div>
  <div class="sec exp"><details><summary>Wie entsteht der Score?</summary><div class="expc"></div></details></div>
  <p class="note">▲ Kauf / ▼ Verkauf = erste Kerze, mit der der Score in die Zone eintritt. Der Score nutzt nur Daten bis zur
  jeweiligen Kerze. Senkrechte gestrichelte Linien = echte Hochs (rot) und Tiefs (grün), die erst im Nachhinein feststehen.
  Datenquelle: <span id="src"></span>. Keine Anlageberatung.</p>
</div>

<script>window.DATA = __DATA__;</script>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
 onerror="var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js';document.head.appendChild(s)"></script>
<script>
(function(){
const D = window.DATA, BUY = D.buy, SELL = D.sell;
const TFL = Object.fromEntries(D.tfs);
const SEC = {'1h':3600,'4h':14400,'1d':86400,'5d':432000,'1w':604800,'2w':1209600,'1M':2629800,'6M':15778800,'1y':31557600};
const $ = id => document.getElementById(id);
const store = (k, v) => { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } };
let filter = 'all', query = '', ltf = store('ltf') || '4h', ctf = store('ctf') || '4h', sortBy = store('sort') || 'mc';
let cur = null, G = null, view = null, liveTs = 0, liveMap = null, reqId = 0, loadingOld = false;
const cache = {};

const color = s => s == null ? '#4b5563' : s >= BUY ? '#22c55e' : s >= 60 ? '#84cc16' : s > 40 ? '#eab308' : s > SELL ? '#f97316' : '#ef4444';
const label = s => s == null ? 'kein Score' : s >= BUY ? 'Starker Kauf' : s >= 60 ? 'Kauf' : s > 40 ? 'Neutral' : s > SELL ? 'Verkauf' : 'Starker Verkauf';
function price(p){
  if (p == null || !isFinite(p)) return '–';
  const a = Math.abs(p), d = a >= 1000 ? 0 : a >= 1 ? 2 : a >= 0.01 ? 4 : 8;
  return '$' + p.toLocaleString('de-DE', {minimumFractionDigits: d, maximumFractionDigits: d});
}
const pct = x => x == null || !isFinite(x) ? '' : (x >= 0 ? '+' : '') + x.toFixed(2).replace('.', ',') + ' %';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const big = () => ['1d','5d','1w','2w','1M','6M','1y'].includes(ctf);
function fmtDate(t){
  const d = new Date(t * 1000);
  if (big()) return d.toLocaleDateString('de-DE', {day: '2-digit', month: '2-digit', year: 'numeric'});
  return d.toLocaleString('de-DE', {day: '2-digit', month: '2-digit', year: '2-digit', hour: '2-digit', minute: '2-digit'});
}
function dur(sec){
  const a = Math.abs(sec), d = Math.floor(a / 86400), h = Math.round((a % 86400) / 3600);
  if (d >= 365) return '≈ ' + (a / 31557600).toFixed(1).replace('.', ',') + ' Jahre';
  if (d >= 60) return '≈ ' + (a / 2629800).toFixed(1).replace('.', ',') + ' Monate';
  if (d >= 1) return d + ' T' + (h ? ' ' + h + ' Std' : '');
  return h + ' Std';
}

// ---------------- Kopfzeile: Markt + Frische
(function(){
  const m = D.mkt || {}, out = [];
  if (m.fng) out.push(`<div class="mb">Fear&Greed <b style="color:${color(100 - m.fng[0])}">${Math.round(m.fng[0])}</b> ${esc(m.fng[1])}</div>`);
  if (m.dom != null) out.push(`<div class="mb">BTC-Dominanz <b>${m.dom.toFixed(1).replace('.', ',')} %</b></div>`);
  if (m.mcap) out.push(`<div class="mb">Marktkap. <b>${(m.mcap / 1e12).toFixed(2).replace('.', ',')} Bio. $</b> <span class="${m.mcap_ch >= 0 ? 'up' : 'dn'}">${pct(m.mcap_ch)}</span></div>`);
  if (m.dxy) out.push(`<div class="mb">DXY <b>${m.dxy.toFixed(1).replace('.', ',')}</b></div>`);
  $('mbar').innerHTML = out.join('');
  if (!out.length) $('mbar').style.display = 'none';
})();
function freshness(){
  const t = Math.max(Date.parse(D.updated), liveTs), min = (Date.now() - t) / 60000;
  const txt = min < 1 ? 'gerade eben' : min < 60 ? `vor ${Math.round(min)} Min` : `vor ${Math.floor(min / 60)} Std ${Math.round(min % 60)} Min`;
  const col = min < 10 ? 'var(--buy)' : min < 70 ? '#eab308' : 'var(--sell)';
  $('ref').innerHTML = `<i style="background:${col}"></i>Preise ${txt} <span>⟳</span>`;
  $('ref2').innerHTML = `<i style="background:${col}"></i>${min < 1 ? 'jetzt' : min < 60 ? Math.round(min) + ' Min' : Math.floor(min / 60) + ' Std'} <span>⟳</span>`;
}
freshness(); setInterval(freshness, 15000);

// ---------------- Liste
$('ltf').insertAdjacentHTML('beforeend', D.ltfs.map(t => `<button class="chip" data-t="${t}">${TFL[t]}</button>`).join(''));
function setLtf(t){ ltf = t; store('ltf', t); document.querySelectorAll('#ltf .chip').forEach(b => b.classList.toggle('on', b.dataset.t === t)); render(); }
function setSort(o){ sortBy = o; store('sort', o); document.querySelectorAll('#sort .chip').forEach(b => b.classList.toggle('on', b.dataset.o === o)); render(); }
$('ltf').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setLtf(b.dataset.t); });
$('sort').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setSort(b.dataset.o); });
function match(c){
  if (query && !(c.s.toLowerCase().includes(query) || c.n.toLowerCase().includes(query))) return false;
  const s = c.sc[ltf];
  if (filter !== 'all' && s == null) return false;
  switch (filter){
    case 'buy': return s >= BUY; case 'pos': return s >= 60 && s < BUY; case 'neu': return s > 40 && s < 60;
    case 'neg': return s > SELL && s <= 40; case 'sell': return s <= SELL;
  }
  return true;
}
function render(){
  const all = D.coins.map(c => c.sc[ltf]).filter(s => s != null);
  const avg = all.length ? all.reduce((a, b) => a + b, 0) / all.length : null;
  $('stats').innerHTML =
    `<div class="stat"><b style="color:${color(avg)}">${avg == null ? '–' : avg.toFixed(0)}</b><span>Ø Score (${TFL[ltf]})</span></div>` +
    `<div class="stat"><b class="up">${all.filter(s => s >= BUY).length}</b><span>Starker Kauf</span></div>` +
    `<div class="stat"><b class="dn">${all.filter(s => s <= SELL).length}</b><span>Starker Verkauf</span></div>`;
  const rows = D.coins.filter(match);
  if (sortBy === 'mc') rows.sort((a, b) => a.r - b.r);
  else rows.sort((a, b) => { const x = a.sc[ltf], y = b.sc[ltf]; if (x == null) return 1; if (y == null) return -1; return sortBy === 'hi' ? y - x : x - y; });
  $('list').innerHTML = rows.length ? rows.map(c => {
    const s = c.sc[ltf], col = color(s);
    return `<li class="row" data-s="${esc(c.s)}"><div class="rk">#${c.r}</div>
      <div class="nm"><b>${esc(c.s)}</b><small>${esc(c.n)}</small>
        <div class="px">${price(c.p)} <span class="${c.ch >= 0 ? 'up' : 'dn'}">${pct(c.ch)}</span> · ${label(s)}</div>
        <div class="bar"><i style="width:${s || 0}%;background:${col}"></i></div></div>
      <div class="sc" style="color:${col};background:${col}22">${s == null ? '–' : Math.round(s)}</div></li>`;
  }).join('') : '<div class="empty">Keine Coins in diesem Filter.</div>';
}
$('q').addEventListener('input', e => { query = e.target.value.trim().toLowerCase(); render(); });
$('chips').addEventListener('click', e => {
  const b = e.target.closest('.chip'); if (!b) return;
  document.querySelectorAll('#chips .chip').forEach(x => x.classList.toggle('on', x === b)); filter = b.dataset.f; render();
});
$('list').addEventListener('click', e => {
  const li = e.target.closest('li.row');
  if (li) { history.pushState({c: li.dataset.s}, '', '#' + encodeURIComponent(li.dataset.s)); openCoin(li.dataset.s); }
});
$('back').addEventListener('click', () => location.hash ? history.back() : closeCoin());
window.addEventListener('popstate', () => { const s = decodeURIComponent(location.hash.slice(1)); s ? openCoin(s) : closeCoin(); });
document.querySelectorAll('#sort .chip').forEach(b => b.classList.toggle('on', b.dataset.o === sortBy));
setLtf(D.ltfs.includes(ltf) ? ltf : '4h');

// ---------------- Erklärung
(function(){
  const names = Object.keys(D.info), grp = g => names.filter(k => D.go[k] === g);
  const li = k => `<li><b>${esc(k)}</b> (Gewicht ${String(D.iw[k] || 1).replace('.', ',')}): ${esc(D.info[k])}</li>`;
  const cfg = D.cal.cfg, r = D.cal.res, d0 = D.cal.default;
  const pctS = x => Math.round(x * 100) + ' %';
  const cal = r ? `<p><b>Auto-Kalibrierung</b> (${new Date(D.cal.date).toLocaleDateString('de-DE')}, wöchentlich):
    Die Gewichte unten wurden an der Historie der Top-Coins (4 Std + 1 T) getestet. Die ersten 70 % der Daten dienen zum
    Einstellen, die letzten 30 % nur zur Kontrolle. ${D.cal.adopted ? 'Die neue Einstellung war auch im Kontrollzeitraum besser und wurde übernommen.' : 'Im Kontrollzeitraum war keine Einstellung besser als der Standard – daher gilt der Standard.'}
    Trefferquote im Kontrollzeitraum: <b>${Math.round(r.test.hit)} %</b> (${r.test.n} Signale)${d0 ? `, Standard: ${Math.round(d0.test.hit)} %` : ''}.</p>` : '';
  const html = `
  <p>Jeder Indikator wird für jede Kerze in einen <b>Teil-Score von 0 bis 100</b> umgerechnet (100 = spricht für Kaufen,
  0 = für Verkaufen). Daraus entstehen drei Gruppen:</p>
  <p><b>1. Lage</b> – Wie weit unten oder oben steht der Kurs?</p><ul>${grp('Lage').map(li).join('')}</ul>
  <p><b>2. Markt</b> (Gewicht ${pctS(cfg.wm / (1 + cfg.wm))} der Richtung) – Stimmung und Umfeld:</p><ul>${grp('Markt').map(li).join('')}</ul>
  <p><b>3. Wende</b> – Gibt es Zeichen, dass der Kurs dreht?</p><ul>${grp('Wende').map(li).join('')}</ul>
  <p><b>So werden sie verbunden:</b> Lage und Markt bestimmen die <b>Richtung</b>. Die Wende ist die <b>Bestätigung</b>:
  Liegt der Kurs weit unten und dreht schon, zählt das Signal voll. Fällt er noch („fallendes Messer“), wird es auf
  ${pctS(Math.max(0.05, 2 * cfg.gb - 1))}–${pctS(cfg.gb)} gedämpft.</p>
  <p><b>Filter:</b> Danach wird das Signal mit dem <b>übergeordneten Zeitrahmen</b> (z. B. 4 Std → 1 T) und dem
  <b>Gesamtmarkt</b> (gewichteter Index der Top-20-Coins) abgeglichen: Zeigen sie in dieselbe Richtung, wird es
  verstärkt, sonst abgeschwächt (Stärke ${String(cfg.kh).replace('.', ',')} bzw. ${String(cfg.km).replace('.', ',')}).
  Es zählt immer nur die zuletzt <i>abgeschlossene</i> übergeordnete Kerze.</p>
  <p><b>Spreizung:</b> Das Ergebnis wird so gestreckt (Faktor ${String(D.cal.S.toFixed(2)).replace('.', ',')}), dass rund 4 % der Kerzen in einer Zone liegen:
  ≥ ${BUY} = Kaufzone (▲), ≤ ${SELL} = Verkaufszone (▼).</p>${cal}
  <p><b>Ehrlichkeit:</b> Für jede Kerze werden nur Daten bis zu dieser Kerze benutzt – alte Signale ändern sich nicht.
  Die gestrichelten Hoch/Tief-Linien stehen erst im Nachhinein fest (±10 Kerzen) und zeigen nur, wie gut der Score getroffen hätte.
  Ältere 1-Std- und 4-Std-Kerzen, die beim Zurückschieben nachgeladen werden, haben keinen Score.</p>
  <p>Kein Indikator kann Tiefs und Hochs sicher vorhersagen. Keine Anlageberatung.</p>`;
  document.querySelectorAll('.expc').forEach(el => el.innerHTML = html);
})();

// ---------------- Aktualisieren (Live von Binance im Browser)
const HOSTS = ['https://data-api.binance.vision', 'https://api.binance.com'];
async function bget(path){
  let err;
  for (const h of HOSTS){
    try { const r = await fetch(h + path, {cache: 'no-store'}); if (r.ok) return await r.json(); err = r.status; if (r.status === 400) break; }
    catch (e) { err = e; }
  }
  throw new Error(err);
}
async function refresh(){
  const btns = [$('ref'), $('ref2')]; btns.forEach(b => b.classList.add('spin'));
  try {
    try {
      const m = await (await fetch('meta.json?t=' + Date.now(), {cache: 'no-store'})).json();
      if (m.updated && m.updated !== D.updated) { location.reload(); return; }
    } catch (e) {}
    const all = await bget('/api/v3/ticker/24hr?type=MINI');
    liveMap = {}; all.forEach(t => liveMap[t.symbol] = t);
    D.coins.forEach(c => { const t = liveMap[c.s + 'USDT']; if (!t) return;
      const last = +t.lastPrice, open = +t.openPrice; c.p = last; if (open) c.ch = (last / open - 1) * 100; });
    liveTs = Date.now(); freshness(); render();
    if (cur) await liveChart();
  } catch (e) {
    $('ref').innerHTML = '⚠️ Live-Preise nicht abrufbar <span>⟳</span>';
  } finally { btns.forEach(b => b.classList.remove('spin')); }
}
$('ref').addEventListener('click', refresh);
$('ref2').addEventListener('click', refresh);

const NATIVE = ['1h', '4h', '1d', '1w', '1M'];
function mergeRows(T, rows){
  rows.forEach(row => {
    const lastT = T.c[T.c.length - 1][0], t = row[0];
    if (t === lastT) T.c[T.c.length - 1] = row;
    else if (t > lastT) { T.c.push(row); if (T.z) T.z.push(null); }
  });
}
async function liveChart(){
  const meta = cur.meta, T = cur.data.tf[ctf]; if (!T) return;
  const sym = meta.s + 'USDT', atEnd = view && view.to >= T.c.length - 2;
  try {
    if (NATIVE.includes(ctf)){
      const k = await bget(`/api/v3/klines?symbol=${sym}&interval=${ctf}&limit=6`);
      mergeRows(T, k.map(r => [Math.floor(r[0] / 1000), +r[1], +r[2], +r[3], +r[4]]));
    } else {
      const t = liveMap && liveMap[sym];
      const p = t ? +t.lastPrice : +(await bget(`/api/v3/ticker/price?symbol=${sym}`)).price;
      const L = T.c[T.c.length - 1]; L[4] = p; L[2] = Math.max(L[2], p); L[3] = Math.min(L[3], p);
    }
    meta.p = T.c[T.c.length - 1][4];
    header();
    if (atEnd) view.to = T.c.length - 1 + 4;
    show(true);
  } catch (e) { $('mres').textContent = 'Live-Kerzen für ' + meta.s + ' nicht bei Binance verfügbar.'; }
}
async function loadOlder(){
  const T = cur && cur.data.tf[ctf];
  if (!T || loadingOld || T.noMore || !['1h', '4h'].includes(ctf)) return;
  loadingOld = true; $('mres').innerHTML = '<span style="color:var(--acc)">Lade ältere Kerzen…</span>';
  try {
    const first = T.c[0][0];
    const k = await bget(`/api/v3/klines?symbol=${cur.meta.s}USDT&interval=${ctf}&endTime=${first * 1000 - 1}&limit=1000`);
    const rows = k.map(r => [Math.floor(r[0] / 1000), +r[1], +r[2], +r[3], +r[4]]).filter(r => r[0] < first);
    if (!rows.length) T.noMore = true;
    else {
      T.c = rows.concat(T.c); if (T.z) T.z = new Array(rows.length).fill(null).concat(T.z);
      T.pre = (T.pre || 0) + rows.length; view.to += rows.length; show(true);
    }
    $('mres').innerHTML = rows.length ? `${rows.length} ältere Kerzen geladen (ohne Score)` : 'Anfang der Historie erreicht.';
  } catch (e) { T.noMore = true; $('mres').textContent = 'Ältere Kerzen nicht verfügbar.'; }
  loadingOld = false;
}

// ---------------- Detail
function destroy(){ if (G){ G.pc.remove(); if (G.zc) G.zc.remove(); } G = null; }
function closeCoin(){ destroy(); cur = null; $('detail').classList.remove('open'); document.body.style.overflow = ''; }
$('tfc').innerHTML = D.tfs.map(([k, l]) => `<button class="chip" data-t="${k}">${l}</button>`).join('');
$('tfc').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b && !b.disabled) { ctf = b.dataset.t; store('ctf', ctf); view = null; M = null; show(); } });

function header(){
  const meta = cur.meta;
  $('dt').textContent = meta.s + ' · ' + meta.n;
  $('ds').innerHTML = price(meta.p) + ' <span class="' + (meta.ch >= 0 ? 'up' : 'dn') + '">' + pct(meta.ch) + '</span> · #' + meta.r;
}
async function openCoin(sym){
  const meta = D.coins.find(x => x.s === sym); if (!meta) return closeCoin();
  destroy(); view = null; M = null;
  $('detail').classList.add('open'); $('detail').scrollTop = 0; document.body.style.overflow = 'hidden';
  cur = {meta, data: null}; header(); $('src').textContent = meta.src; $('mres').textContent = 'Lade Daten…';
  const id = ++reqId;
  try {
    if (!cache[meta.f]) {
      const r = await fetch('data/' + meta.f + '.json?v=' + encodeURIComponent(D.updated));
      if (!r.ok) throw new Error(r.status);
      cache[meta.f] = await r.json();
    }
  } catch (e) { if (id === reqId) $('mres').textContent = 'Daten konnten nicht geladen werden.'; return; }
  if (id !== reqId) return;
  cur.data = cache[meta.f];
  show();
}
function waitLib(cb, n){ if (window.LightweightCharts) cb(); else if ((n || 0) < 100) setTimeout(() => waitLib(cb, (n || 0) + 1), 100); }

function show(keep){
  if (!cur || !cur.data) return;
  const tfs = cur.data.tf;
  if (!tfs[ctf]) ctf = tfs['4h'] ? '4h' : Object.keys(tfs)[0];
  document.querySelectorAll('#tfc .chip').forEach(b => { b.disabled = !tfs[b.dataset.t]; b.classList.toggle('on', b.dataset.t === ctf); });
  const T = tfs[ctf], hasZ = !!T.z;
  const s = hasZ ? T.z.slice().reverse().find(x => x != null) : null, col = color(s);
  $('dsc').textContent = s == null ? '–' : Math.round(s); $('dsc').style.color = col; $('dsc').style.background = col + '22';
  ['fbox', 'catbox', 'pcbox', 'allbox', 'statbox'].forEach(id => $(id).style.display = hasZ && T.b ? '' : 'none');
  $('zcw').style.display = hasZ ? '' : 'none'; $('noz').style.display = hasZ ? 'none' : '';
  if (hasZ && T.b) details(T);
  if (!keep) M = null;
  destroy();
  waitLib(() => draw(T));
}

function details(T){
  const st = T.stat || {}, H = D.h;
  const box = (k, name) => { const v = st[k];
    if (!v) return `<div class="cat">${name}<b>–</b>keine Signale</div>`;
    return `<div class="cat">${name} (${v[0]}×)<b style="color:${v[2] >= 50 ? 'var(--buy)' : 'var(--sell)'}">${v[2]} % Treffer</b>Ø ${pct(v[1])} nach ${H} Kerzen</div>`; };
  $('stat').innerHTML = box('buy', '▲ Kauf-Signale') + box('sell', '▼ Verkaufs-Signale');
  const f = T.f || {};
  $('filt').innerHTML = (f.htf ? `<div class="cat">Übergeordnet (${TFL[f.htf[0]]})<b style="color:${color(f.htf[1])}">${f.htf[1] ?? '–'}</b>${label(f.htf[1])}</div>` : '<div class="cat">Übergeordnet<b>–</b>nicht vorhanden</div>') +
    `<div class="cat">Gesamtmarkt (Top 20)<b style="color:${color(f.mkt)}">${f.mkt ?? '–'}</b>${label(f.mkt)}</div>`;
  $('cats').innerHTML = ['Lage', 'Markt', 'Wende'].map(k => {
    const v = T.k[k]; if (v == null) return `<div class="cat">${k}<b>–</b></div>`;
    return `<div class="cat">${k}<b style="color:${color(v)}">${v}</b><div class="bar"><i style="width:${v}%;background:${color(v)}"></i></div></div>`;
  }).join('');
  const sorted = [...T.b].sort((a, b) => b[2] - a[2]);
  const item = x => `<div class="it"><b><span style="color:var(--txt)">${esc(x[0])}</span><span style="color:${color(x[2])}">${x[2]}</span></b><span>${esc(x[4])}</span></div>`;
  const pro = sorted.filter(x => x[2] >= 60).slice(0, 4), con = sorted.filter(x => x[2] <= 40).reverse().slice(0, 4);
  $('pro').innerHTML = pro.length ? pro.map(item).join('') : '<div class="it"><span>Keine deutlichen Kaufsignale</span></div>';
  $('con').innerHTML = con.length ? con.map(item).join('') : '<div class="it"><span>Keine deutlichen Verkaufssignale</span></div>';
  $('all').innerHTML = ['Lage', 'Markt', 'Wende'].map(g => `<tr><td colspan="3" style="color:var(--mut);font-weight:600">${g}</td></tr>` +
    T.b.filter(x => x[1] === g).map(x => `<tr><td>${esc(x[0])}<br><span style="color:var(--mut);font-size:11px">${esc(x[4])}</span></td>
     <td class="v">${esc(x[3])}</td><td class="v"><span class="mini"><i style="width:${x[2]}%;background:${color(x[2])}"></i></span> <b style="color:${color(x[2])}">${x[2]}</b></td></tr>`).join('')).join('');
}

const AX = 44;   // Höhe der Zeitachse (zum Schieben/Zoomen)
function draw(T){
  const L = window.LightweightCharts;
  const hasZ = !!T.z;
  const base = {
    autoSize: true, handleScroll: false, handleScale: false,
    layout: {background: {type: 'solid', color: '#131821'}, textColor: '#8b93a5', fontSize: 11},
    grid: {vertLines: {color: '#1c2230'}, horzLines: {color: '#1c2230'}},
    rightPriceScale: {borderColor: '#242b38', minimumWidth: 78},
    timeScale: {borderColor: '#242b38', timeVisible: !big(), secondsVisible: false, rightOffset: 0, minBarSpacing: 0.05,
      minimumHeight: AX, lockVisibleTimeRangeOnResize: true,
      tickMarkFormatter: (t, type) => { const d = new Date(t * 1000);
        if (type === 0) return String(d.getFullYear());
        if (type === 1) return d.toLocaleDateString('de-DE', {month: 'short'});
        if (type === 2) return d.getDate() + '.' + (d.getMonth() + 1) + '.';
        return d.toLocaleTimeString('de-DE', {hour: '2-digit', minute: '2-digit'}); }},
    crosshair: {mode: L.CrosshairMode.Normal, vertLine: {color: '#60a5fa', labelBackgroundColor: '#2563eb'},
      horzLine: {color: '#60a5fa', labelBackgroundColor: '#2563eb'}},
    localization: {locale: 'de-DE', timeFormatter: t => fmtDate(t)},
  };
  // Mit Score-Chart: Zeitachse nur ganz unten, damit die Hoch/Tief-Linien ohne Lücke durchlaufen
  $('pc').style.height = hasZ ? '330px' : (330 + AX) + 'px';
  $('ptouch').style.bottom = hasZ ? '0' : AX + 'px';
  $('ztouch').style.bottom = AX + 'px';
  const n = T.c.length, last = T.c[n - 1][4];
  const prec = last >= 1000 ? 2 : last >= 1 ? 3 : last >= 0.01 ? 5 : 8;
  const pc = L.createChart($('pc'), Object.assign({}, base, {timeScale: Object.assign({}, base.timeScale, {visible: !hasZ})}));
  const cs = pc.addCandlestickSeries({upColor: '#22c55e', downColor: '#ef4444', borderVisible: false,
    wickUpColor: '#22c55e', wickDownColor: '#ef4444', priceFormat: {type: 'price', precision: prec, minMove: Math.pow(10, -prec)}});
  cs.setData(T.c.map(k => ({time: k[0], open: k[1], high: k[2], low: k[3], close: k[4]})));
  const markers = [];
  if (hasZ) for (let i = 1; i < T.z.length; i++){
    const s = T.z[i], p = T.z[i - 1]; if (s == null) continue;
    if (s >= BUY && (p == null || p < BUY)) markers.push({time: T.c[i][0], position: 'belowBar', color: '#22c55e', shape: 'arrowUp', text: 'Kauf'});
    if (s <= SELL && (p == null || p > SELL)) markers.push({time: T.c[i][0], position: 'aboveBar', color: '#ef4444', shape: 'arrowDown', text: 'Verkauf'});
  }
  cs.setMarkers(markers);
  let zc = null, zs = null;
  if (hasZ){
    zc = L.createChart($('zc'), base);
    zs = zc.addBaselineSeries({baseValue: {type: 'price', price: 50}, lineWidth: 2,
      topLineColor: '#22c55e', topFillColor1: 'rgba(34,197,94,.28)', topFillColor2: 'rgba(34,197,94,.02)',
      bottomLineColor: '#ef4444', bottomFillColor1: 'rgba(239,68,68,.02)', bottomFillColor2: 'rgba(239,68,68,.28)',
      priceFormat: {type: 'price', precision: 0, minMove: 1}, lastValueVisible: true, priceLineVisible: false,
      autoscaleInfoProvider: () => ({priceRange: {minValue: 0, maxValue: 100}})});
    zs.setData(T.c.map((k, i) => T.z[i] == null ? {time: k[0]} : {time: k[0], value: T.z[i]}));
    zs.createPriceLine({price: BUY, color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Kauf'});
    zs.createPriceLine({price: SELL, color: '#ef4444', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Verkauf'});
    zs.createPriceLine({price: 50, color: '#4b5563', lineWidth: 1, lineStyle: 3, axisLabelVisible: false});
  }
  G = {pc, cs, zc, zs, T, n};
  if (!view) view = {bars: Math.min(n, big() ? 150 : 120), to: n - 1 + 4};
  clampView(); applyView();
  // Preisachsen beider Charts gleich breit machen, damit die Linien exakt übereinander liegen
  requestAnimationFrame(() => {
    if (!G || !G.zc) return;
    try {
      const w = Math.max(G.pc.priceScale('right').width(), G.zc.priceScale('right').width(), 78);
      G.pc.applyOptions({rightPriceScale: {minimumWidth: w}}); G.zc.applyOptions({rightPriceScale: {minimumWidth: w}});
      requestAnimationFrame(() => { drawPiv(); drawM(); });
    } catch (e) {}
  });
}

// ---------------- Ansicht: Schieben und Zoomen auf der Zeitachse
const MINB = 10;
const maxBars = () => Math.max(MINB + 1, G.n + 8);
function plotWidth(){
  let w = 78; try { w = G.pc.priceScale('right').width(); } catch (e) {}
  return Math.max(50, $('pc').clientWidth - w);
}
function clampView(){
  view.bars = Math.max(MINB, Math.min(maxBars(), view.bars));
  view.to = Math.min(G.n - 1 + 8, Math.max(Math.min(view.bars - 1, G.n + 7), view.to));
}
function applyView(){
  if (!G) return;
  const r = {from: view.to - view.bars, to: view.to};
  G.pc.timeScale().setVisibleLogicalRange(r); if (G.zc) G.zc.timeScale().setVisibleLogicalRange(r);
  requestAnimationFrame(() => { drawPiv(); drawM(); });
  if (view.to - view.bars < 15) loadOlder();
}
$('now').addEventListener('click', () => { if (!G) return; view.to = G.n - 1 + 4; clampView(); applyView(); });
function cap(el, e){ try { el.setPointerCapture(e.pointerId); } catch (x) {} }
const ax = $('axt'), axP = new Map();
let axStart = null;
function axBegin(){
  const pts = [...axP.values()], r = $('pc').getBoundingClientRect();
  if (!pts.length) { axStart = null; return; }
  const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length - r.left;
  axStart = {to: view.to, bars: view.bars, mx, d: pts.length > 1 ? Math.abs(pts[0].x - pts[1].x) : 0, n: pts.length};
}
ax.addEventListener('pointerdown', e => { if (!G) return; e.preventDefault(); cap(ax, e); axP.set(e.pointerId, {x: e.clientX}); axBegin(); });
ax.addEventListener('pointermove', e => {
  if (!G || !axStart || !axP.has(e.pointerId)) return;
  axP.set(e.pointerId, {x: e.clientX});
  const pts = [...axP.values()], r = $('pc').getBoundingClientRect(), W = plotWidth();
  const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length - r.left;
  let bars = axStart.bars;
  if (axStart.n > 1 && pts.length > 1) {
    const d = Math.abs(pts[0].x - pts[1].x);
    if (axStart.d > 10 && d > 10) bars = axStart.bars * axStart.d / d;
  }
  // Der Punkt unter den Fingern bleibt unter den Fingern (wie beim Chart selbst)
  const fromStart = axStart.to - axStart.bars, anchor = fromStart + axStart.mx / W * axStart.bars;
  bars = Math.max(MINB, Math.min(maxBars(), bars));
  const from = anchor - mx / W * bars;
  view.bars = bars; view.to = from + bars; clampView(); applyView();
});
['pointerup', 'pointercancel'].forEach(ev => ax.addEventListener(ev, e => { axP.delete(e.pointerId); axBegin(); }));
ax.addEventListener('wheel', e => { if (!G) return; e.preventDefault(); view.bars *= e.deltaY > 0 ? 1.15 : 0.87; clampView(); applyView(); }, {passive: false});
window.addEventListener('resize', () => { if (G) applyView(); });

// ---------------- Hoch/Tief-Linien – dauerhaft, durchgehend durch Kurs- und Score-Chart
function drawPiv(){
  const el = $('pvl'); el.innerHTML = '';
  if (!G || !G.T.p) return;
  el.style.bottom = AX + 'px';
  const ts = G.pc.timeScale(), lines = [];
  for (const [t, k] of G.T.p) {
    const x = ts.timeToCoordinate(t); if (x == null) continue;
    lines.push(`<div class="vl" style="left:${x}px;border-color:${k === 'H' ? 'rgba(239,68,68,.55)' : 'rgba(34,197,94,.55)'}"></div>`);
    if (lines.length > 400) break;
  }
  el.innerHTML = lines.join('');
}

// ---------------- Messen: 1 Finger = Start fest, Ende folgt; 2 Finger = jeder Finger hält einen Punkt
let M = null;
const fingers = new Map();   // pointerId -> 'a' | 'b'
function timeOf(l){
  if (l >= 0 && l < G.n) return G.T.c[Math.round(l)][0];
  const step = SEC[ctf];
  return l < 0 ? G.T.c[0][0] + l * step : G.T.c[G.n - 1][0] + (l - G.n + 1) * step;
}
function pointAt(e){
  const r = $('pc').getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const l = G.pc.timeScale().coordinateToLogical(x), p = G.cs.coordinateToPrice(y);
  if (l == null || p == null) return null;
  return {l: Math.round(l), p};
}
function drawM(){
  const layer = $('ml'); layer.innerHTML = '';
  if (!G || !M) { if (G) { G.pc.clearCrosshairPosition(); if (G.zc) G.zc.clearCrosshairPosition(); } return; }
  const ts = G.pc.timeScale(), pos = q => ({x: ts.logicalToCoordinate(q.l), y: G.cs.priceToCoordinate(q.p)});
  const B = M.b, idx = Math.max(0, Math.min(G.n - 1, B.l)), inData = B.l >= 0 && B.l < G.n;
  if (inData) {
    G.pc.setCrosshairPosition(B.p, G.T.c[idx][0], G.cs);
    if (G.zc && G.T.z[idx] != null) G.zc.setCrosshairPosition(G.T.z[idx], G.T.c[idx][0], G.zs); else if (G.zc) G.zc.clearCrosshairPosition();
  } else G.pc.clearCrosshairPosition();
  const pb = pos(B), W = $('pc').clientWidth; let html = '', res = '';
  if (M.moved) {
    const pa = pos(M.a), lo = M.a.l <= B.l ? M.a : B, hi = M.a.l <= B.l ? B : M.a;
    const up = hi.p >= lo.p, col = up ? '#22c55e' : '#ef4444';
    const dp = hi.p - lo.p, pc_ = dp / lo.p * 100, bars = hi.l - lo.l, d = dur(bars * SEC[ctf]);
    if (pa.x != null && pb.x != null && pa.y != null && pb.y != null)
      html += `<div class="rect" style="left:${Math.min(pa.x, pb.x)}px;top:${Math.min(pa.y, pb.y)}px;width:${Math.abs(pb.x - pa.x)}px;height:${Math.abs(pb.y - pa.y)}px;border-color:${col};background:${col}22"></div>
        <div class="dot" style="left:${pa.x}px;top:${pa.y}px;background:#60a5fa"></div><div class="dot" style="left:${pb.x}px;top:${pb.y}px;background:${col}"></div>`;
    res = `<b style="color:${col}">${pct(pc_)}</b> (${dp >= 0 ? '+' : '−'}${price(Math.abs(dp))}) · ${bars} Kerzen · ${d}`;
    if (pa.x != null && pb.x != null) {
      const cx = Math.min(Math.max((pa.x + pb.x) / 2, 90), W - 90), top = Math.max(64, Math.min(pa.y ?? 0, pb.y ?? 0) - 14);
      html += `<div class="bub" style="left:${cx}px;top:${top}px"><b style="color:${col}">${pct(pc_)}</b> · ${d}<br>${price(lo.p)} → ${price(hi.p)}<br>${fmtDate(timeOf(lo.l))} → ${fmtDate(timeOf(hi.l))}</div>`;
    }
  } else if (pb.x != null && pb.y != null) {
    const top = Math.max(52, pb.y - 18), left = Math.min(Math.max(pb.x, 70), W - 70);
    html += `<div class="bub" style="left:${left}px;top:${top}px"><b>${price(B.p)}</b><br>${fmtDate(timeOf(B.l))}</div>`;
    res = `<span style="color:var(--acc)">${price(B.p)} · ${fmtDate(timeOf(B.l))}</span>`;
  }
  layer.innerHTML = html; $('mres').innerHTML = res;
}
const pt = $('ptouch');
pt.addEventListener('pointerdown', e => {
  if (!G) return; e.preventDefault(); cap(pt, e);
  const q = pointAt(e); if (!q) return;
  if (fingers.size === 0 || !M) {             // neuer Messvorgang
    fingers.clear(); fingers.set(e.pointerId, 'b');
    M = {a: q, b: q, moved: false, x0: e.clientX, y0: e.clientY};
  } else if (fingers.size === 1) {            // zweiter Finger übernimmt den Startpunkt
    fingers.set(e.pointerId, 'a'); M.a = q; M.moved = true;
  }
  drawM();
});
pt.addEventListener('pointermove', e => {
  if (!G) return;
  const q = pointAt(e); if (!q) return;
  const role = fingers.get(e.pointerId);
  if (!role) { if (e.pointerType === 'mouse' && fingers.size === 0) { M = {a: q, b: q, moved: false, hover: true}; drawM(); } return; }
  M[role] = q;
  if (!M.moved && Math.hypot(e.clientX - M.x0, e.clientY - M.y0) > 8) M.moved = true;
  drawM();
});
['pointerup', 'pointercancel'].forEach(ev => pt.addEventListener(ev, e => fingers.delete(e.pointerId)));
pt.addEventListener('pointerleave', () => { if (M && M.hover) { M = null; drawM(); $('mres').innerHTML = ''; } });
const zt = $('ztouch');
function zMove(e){
  if (!G || !G.zc) return;
  const r = $('zc').getBoundingClientRect(), l = G.zc.timeScale().coordinateToLogical(e.clientX - r.left);
  if (l == null) return; const i = Math.max(0, Math.min(G.n - 1, Math.round(l)));
  M = null; drawM();
  if (G.T.z[i] != null) G.zc.setCrosshairPosition(G.T.z[i], G.T.c[i][0], G.zs);
  G.pc.setCrosshairPosition(G.T.c[i][4], G.T.c[i][0], G.cs);
  $('mres').innerHTML = G.T.z[i] != null ? `<span style="color:${color(G.T.z[i])}">Score ${Math.round(G.T.z[i])}</span> · ${fmtDate(G.T.c[i][0])}` : fmtDate(G.T.c[i][0]) + ' · kein Score';
}
zt.addEventListener('pointerdown', e => { e.preventDefault(); cap(zt, e); zMove(e); });
zt.addEventListener('pointermove', e => { if (e.buttons || e.pointerType !== 'mouse') zMove(e); });

const h0 = decodeURIComponent(location.hash.slice(1));
if (h0) openCoin(h0);
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    try:
        run(html=HTML)
    except KeyboardInterrupt:
        sys.exit(1)
