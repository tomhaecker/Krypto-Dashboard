#!/usr/bin/env python3
"""
Krypto-Dashboard v5 – läuft automatisch über GitHub Actions (stündlich).

Ablauf:
1. Top-Coins nach Marktkapitalisierung (CoinGecko, Ersatz CoinPaprika), ohne Stablecoins/Wrapped.
2. Kerzen in 9 Zeitrahmen (Binance, Ersatz OKX/KuCoin). Tag/Woche/Monat mit kompletter Historie.
3. Markt-Daten: Fear & Greed Index, Dollar-Index (DXY), Gold, BTC-Dominanz (Näherung),
   eigener Gesamtmarkt-Index aus den Top-20-Coins.
4. Antizyklischer Score 0–100 je Kerze (ohne Repainting) in zwei Modi (kurzfristig / langfristig):
   Lage + Markt (Richtung), Wende inkl. Zyklenanalyse (Bestätigung), übergeordneter Zeitrahmen
   und Gesamtmarkt (Filter). Kauf-Signal einmal je Runde, Verkauf in drei Tranchen.
5. Wöchentliche Auto-Kalibrierung: Kaufsignale möglichst selten und möglichst nah am echten Tief,
   geprüft an einem getrennten Testzeitraum.
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
BUY_LEVEL = 80
SELL_LEVELS = (30, 20, 10)   # Verkauf in drei Tranchen
RESET_LEVEL = 50             # neue Signal-Runde erst, wenn der Score wieder die Mitte erreicht
STAT_HORIZON = 10
INDEX_COINS = 20
CALIB_MAX_AGE_DAYS = 7
CALIB_COINS = 40
CALIB_VERSION = 5
OUT_DIR = "site"

LIST_TFS = ["1h", "4h", "1d", "1w"]
TIMEFRAMES = [("1h", "1 Std"), ("4h", "4 Std"), ("1d", "1 T"), ("5d", "5 T"), ("1w", "1 W"),
              ("2w", "2 W"), ("1M", "1 M"), ("6M", "6 M"), ("1y", "1 J")]
TF_SEC = {"1h": 3600, "4h": 14400, "1d": 86400, "5d": 432000, "1w": 604800, "2w": 1209600,
          "1M": 2629800, "6M": 15778800, "1y": 31557600}
HTF = {"1h": "4h", "4h": "1d", "1d": "1w", "5d": "2w", "1w": "1M", "2w": "1M"}
NATIVE_LIMIT = {"1h": INTRADAY_KEEP, "4h": INTRADAY_KEEP}
FULL_HISTORY = {"1d", "1w", "1M"}

GROUP_OF = {
    "RSI (14)": "Lage", "Stochastic RSI": "Lage", "Williams %R": "Lage", "CCI (20)": "Lage", "MFI (14)": "Lage",
    "Bollinger %B": "Lage", "Abstand EMA 50": "Lage", "Abstand SMA 200": "Lage",
    "Lage in 100-Kerzen-Spanne": "Lage", "Lage in 300-Kerzen-Spanne": "Lage",
    "MACD dreht": "Wende", "RSI-Divergenz": "Wende", "Stoch-RSI Kreuzung": "Wende", "RSI-Schwung (3)": "Wende",
    "Erholung vom Tief": "Wende", "Volumen-Klimax": "Wende", "Kerzen-Docht": "Wende", "Zyklen": "Wende",
    "Fear & Greed": "Markt", "Dollar-Index (DXY)": "Markt", "Bitcoin/Gold": "Markt", "BTC-Dominanz": "Markt",
}
BASE_WEIGHTS = {"Lage in 100-Kerzen-Spanne": 2.5, "Lage in 300-Kerzen-Spanne": 0.5, "Abstand EMA 50": 1.5,
                "Abstand SMA 200": 1.5, "Bollinger %B": 1.0, "RSI (14)": 0.75, "Stochastic RSI": 0.5,
                "Williams %R": 0.5, "CCI (20)": 0.5, "MFI (14)": 0.75,
                "Erholung vom Tief": 2.0, "Stoch-RSI Kreuzung": 1.0, "RSI-Schwung (3)": 1.0, "MACD dreht": 1.0,
                "RSI-Divergenz": 1.0, "Volumen-Klimax": 0.5, "Kerzen-Docht": 0.5, "Zyklen": 1.5,
                "Fear & Greed": 1.0, "Dollar-Index (DXY)": 1.0, "Bitcoin/Gold": 1.0, "BTC-Dominanz": 1.0}
MODES = {
    "short": {"label": "Kurzfristig", "pivN": 10,
              "weights": BASE_WEIGHTS,
              "default": {"wm": 0.35, "gb": 0.4, "kh": 0.5, "km": 0.5, "share": 0.02},
              "grid": {"wm": [0.0, 0.35, 0.7], "gb": [0.2, 0.4, 0.6], "kh": [0.0, 0.5, 1.0],
                       "km": [0.0, 0.5, 1.0], "share": [0.01, 0.02, 0.04]}},
    "long": {"label": "Langfristig", "pivN": 30,
             "weights": dict(BASE_WEIGHTS, **{"Lage in 100-Kerzen-Spanne": 1.0, "Lage in 300-Kerzen-Spanne": 3.0,
                                              "Abstand SMA 200": 3.0, "Abstand EMA 50": 1.0, "Stochastic RSI": 0.25,
                                              "Williams %R": 0.25, "CCI (20)": 0.25, "RSI-Schwung (3)": 0.5,
                                              "Kerzen-Docht": 0.25, "Volumen-Klimax": 0.25, "Zyklen": 2.0,
                                              "Fear & Greed": 1.5, "Bitcoin/Gold": 1.5}),
             "default": {"wm": 0.5, "gb": 0.3, "kh": 1.0, "km": 0.5, "share": 0.01},
             "grid": {"wm": [0.25, 0.5, 1.0], "gb": [0.1, 0.3, 0.5], "kh": [0.5, 1.0, 1.5],
                      "km": [0.0, 0.5, 1.0], "share": [0.005, 0.01, 0.02]}},
}
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
    "Lage in 300-Kerzen-Spanne": "Wie oben, aber über 300 Kerzen – findet die großen Tiefs und Hochs.",
    "MACD dreht": "Wird das MACD-Histogramm gerade größer (dreht hoch) oder kleiner (dreht runter)?",
    "RSI-Divergenz": "Kurs macht neues Tief, RSI aber nicht = bullisch. Umgekehrt am Hoch = bärisch.",
    "Stoch-RSI Kreuzung": "Kreuzt die schnelle K-Linie die langsame D-Linie nach oben (Kauf) oder unten?",
    "RSI-Schwung (3)": "Steigt oder fällt der RSI in den letzten 3 Kerzen?",
    "Erholung vom Tief": "Hat sich der Kurs schon vom Tief der letzten 6 Kerzen gelöst (Kauf) bzw. vom Hoch (Verkauf)?",
    "Volumen-Klimax": "Rote Kerze mit sehr hohem Volumen = Ausverkauf (Kauf), grüne = Kaufrausch (Verkauf).",
    "Kerzen-Docht": "Langer unterer Docht = Käufer wehren ab (Kauf), langer oberer = Verkäufer (Verkauf).",
    "Zyklen": "Zyklenanalyse: Aus den letzten bis zu 400 Kerzen werden die 3 Wellenlängen gesucht, die am besten zum "
              "Kursverlauf passen, und nach vorne gerechnet. Zyklus nahe Tief = Kauf, nahe Hoch = Verkauf. "
              "Alle 10 Kerzen neu berechnet, nur mit Daten bis dahin.",
    "Fear & Greed": "Stimmung am Kryptomarkt (alternative.me). Extreme Angst = Kauf, extreme Gier = Verkauf.",
    "Dollar-Index (DXY)": "Stärke des US-Dollars. Steigender Dollar bremst Krypto (Verkauf), fallender hilft (Kauf).",
    "Bitcoin/Gold": "Bitcoin-Preis in Unzen Gold, verglichen mit dem Durchschnitt des letzten Jahres. Billig = Kauf.",
    "BTC-Dominanz": "Nur Altcoins. Steigt Bitcoin gegenüber den Altcoins, fließt Geld ab (Verkauf), fällt sie = Kauf. "
                    "Näherung aus den Top-20-Coins.",
}


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
SESSION.headers["User-Agent"] = "krypto-dashboard/5.0 (+github-actions)"
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
    """Tägliche Markt-Indikatoren: {name: (verfügbar_ab, score, rohwert, erklärung, format, anzeige)}."""
    out = {}
    if "fng" in m:
        s, lag = m["fng"]
        out["Fear & Greed"] = (s.index.values + lag, lin(s, 15, 85).values, s.values,
                               lambda v: ("extreme Angst" if v < 25 else "Angst" if v < 45 else "neutral" if v <= 55
                                          else "Gier" if v <= 75 else "extreme Gier"), lambda v: f"{v:.0f}", s.values)
    if "dxy" in m:
        s, lag = m["dxy"]
        z = (s / ema(s, 50) - 1) / 0.015
        out["Dollar-Index (DXY)"] = (s.index.values + lag, (50 - 50 * np.tanh(z / 1.5)).values, z.values,
                                     lambda v: "Dollar stark (Gegenwind)" if v > 0.5 else "Dollar schwach (Rückenwind)"
                                     if v < -0.5 else "neutral", lambda v: f"{v:+.1f}", s.values)
    if "gold" in m and btc_daily is not None and len(btc_daily) > 400:
        g, lag = m["gold"]
        bt = pd.Series(btc_daily["c"].values, index=(btc_daily["t"].values // 86400) * 86400)
        gd = g.reindex(bt.index, method="ffill")
        ratio = bt / gd
        lr = np.log(ratio)
        z = (lr - lr.rolling(365, min_periods=200).mean()) / lr.rolling(365, min_periods=200).std()
        out["Bitcoin/Gold"] = (bt.index.values + 86400, (50 - 50 * np.tanh(z / 1.5)).values, z.values,
                               lambda v: "BTC günstig ggü. Gold" if v < -0.75 else "BTC teuer ggü. Gold" if v > 0.75
                               else "normal", lambda v: f"{v:+.1f} σ", ratio.values)
    return out


def cycles(c, win=400, top=3):
    """Zyklenanalyse ohne Blick in die Zukunft: alle K Kerzen werden die stärksten Wellenlängen
    der Vergangenheit bestimmt und bis zur nächsten Neuberechnung fortgeschrieben. Wert -1 (Tief) … +1 (Hoch)."""
    y = np.log(np.asarray(c, dtype=float))
    n = len(y)
    out = np.full(n, np.nan)
    last_per = []
    if n < 120:
        return out, last_per
    K = max(10, n // 150)
    for i in range(119, n, K):
        W = min(win, i + 1)
        seg = y[i - W + 1:i + 1]
        t = np.arange(W)
        A = np.vstack([t, np.ones(W)]).T
        r = seg - A @ np.linalg.lstsq(A, seg, rcond=None)[0]
        P = np.unique(np.round(np.geomspace(8, max(16, W / 2.5), 48)))
        ang = 2 * np.pi * t[None, :] / P[:, None]
        a = (r[None, :] * np.sin(ang)).sum(1) * 2 / W
        b = (r[None, :] * np.cos(ang)).sum(1) * 2 / W
        amp = np.hypot(a, b)
        peaks = [j for j in range(len(P)) if (j == 0 or amp[j] >= amp[j - 1]) and (j == len(P) - 1 or amp[j] >= amp[j + 1])]
        peaks = sorted(peaks, key=lambda j: -amp[j])[:top]
        if not peaks:
            continue
        jj = np.arange(i, min(n, i + K))
        tt = jj - (i - W + 1)
        val = np.zeros(len(jj))
        for p in peaks:
            w = 2 * np.pi * tt / P[p]
            val += a[p] * np.sin(w) + b[p] * np.cos(w)
        out[jj] = np.clip(val / max(amp[peaks].sum(), 1e-9), -1, 1)
        last_per = [[int(P[p]), round(float(amp[p] * 100), 1)] for p in peaks]
    return out, last_per


def with_groups(C, weights):
    """Gruppen Lage/Wende/Markt mit den Gewichten eines Modus."""
    subs = C["subs"]
    g = {}
    for grp in ("Lage", "Wende", "Markt"):
        cols = [x for x in subs.columns if GROUP_OF.get(x) == grp]
        if not cols:
            g[grp] = np.full(len(subs), np.nan)
            continue
        w = np.array([weights.get(x, 1.0) for x in cols])
        part = subs[cols].values
        mask = ~np.isnan(part)
        with np.errstate(invalid="ignore", divide="ignore"):
            g[grp] = np.where(mask.any(axis=1), np.nansum(part * w, axis=1) / (mask * w).sum(axis=1), np.nan)
    return dict(C, lage=g["Lage"], wende=g["Wende"], markt=g["Markt"])


def combine(C, cfg, htf_base=None, mkt_base=None):
    """Score-Formel. Gibt (Basis ohne Filter, Rohwert) zurück, beide etwa -1..1."""
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
                f = f * np.clip(1 + 0.6 * kk * a, 0.25, 1.6)
    raw = base * f
    base = base.copy()
    base[:30] = np.nan
    raw[:30] = np.nan
    return base, raw


def to_score(raw, S):
    return np.clip(50 + 50 * raw * S, 0, 100)


def htf_map(C_low, C_high):
    return np.searchsorted(C_high["ct"], C_low["ct"], side="right") - 1


def exact_index(src_t, t):
    pos = np.searchsorted(src_t, t)
    pos_c = np.clip(pos, 0, len(src_t) - 1)
    return np.where(np.asarray(src_t)[pos_c] == t, pos_c, -1)


def take(arr, idx):
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = arr[idx[ok]]
    return out


def pivot_idx(h, l, n):
    hs, ls = pd.Series(h), pd.Series(l)
    w = 2 * n + 1
    hi = (hs == hs.rolling(w, center=True).max()).fillna(False).values
    lo = (ls == ls.rolling(w, center=True).min()).fillna(False).values
    return np.where(hi)[0], np.where(lo)[0]


def signals(score):
    """Kauf: erster Eintritt in ≥80 je Runde. Verkauf: erster Eintritt in ≤30 / ≤20 / ≤10 je Runde.
    Eine neue Runde beginnt erst, wenn der Score wieder die Mitte (50) erreicht."""
    s = np.asarray(score, dtype=float)
    valid = ~np.isnan(s)
    sv = np.where(valid, s, 50.0)
    prev = np.r_[50.0, sv[:-1]]

    def first_in_round(enter, reset):
        idx = np.where(enter & valid)[0]
        rs = np.where(reset & valid)[0]
        out, last = [], -1
        for i in idx:
            if last < 0 or (np.searchsorted(rs, i) - np.searchsorted(rs, last + 1)) > 0:
                out.append(int(i))
                last = i
        return out

    buys = first_in_round((sv >= BUY_LEVEL) & (prev < BUY_LEVEL), sv <= RESET_LEVEL)
    sells = []
    for k, lv in enumerate(SELL_LEVELS, 1):
        sells.append(first_in_round((sv <= lv) & (prev > lv), sv >= RESET_LEVEL))
    return buys, sells


# ----------------------------------------------------------------- Kalibrierung
def nearness(sig_idx, piv_idx, n_total, N):
    """1 = Signal genau am echten Tief/Hoch, 0 = N Kerzen oder weiter entfernt."""
    out = []
    for i in sig_idx:
        if i >= n_total - N or not len(piv_idx):
            continue
        d = np.min(np.abs(piv_idx - i))
        out.append(max(0.0, 1 - d / N))
    return out


def evaluate(items, cfg, mode):
    N = MODES[mode]["pivN"]
    cache = {}
    raws = []
    for it in items:
        hb = mb = None
        if it["H"] is not None:
            k = id(it["H"])
            if k not in cache:
                cache[k] = combine(it["H"], cfg)[0]
            hb = take(cache[k], it["hidx"])
        if it["M"] is not None:
            k = id(it["M"])
            if k not in cache:
                cache[k] = combine(it["M"], cfg)[0]
            mb = take(cache[k], it["midx"])
        raws.append(combine(it["C"], cfg, hb, mb)[1])
    pool = np.concatenate([np.abs(r[it["train"]]) for r, it in zip(raws, items)])
    pool = pool[~np.isnan(pool)]
    q = np.quantile(pool, 1 - cfg["share"]) if len(pool) > 100 else 0.3
    S = float(np.clip(0.6 / max(q, 1e-6), 0.8, 6.0))
    res = {}
    for part in ("train", "test"):
        bn, sn, rets, lows = [], [], [], 0
        for r, it in zip(raws, items):
            sc = to_score(r, S)
            mask = it[part]
            buys, sells = signals(np.where(mask, sc, np.nan))
            n = len(sc)
            bn += nearness(buys, it["lo"][N], n, N)
            sn += nearness(sells[0], it["hi"][N], n, N)
            lo_in = it["lo"][N]
            lows += int(np.sum(mask[lo_in])) if len(lo_in) else 0
            c = it["C"]["c"]
            rets += [c[i + STAT_HORIZON] / c[i] - 1 for i in buys if i + STAT_HORIZON < n]
        nb = len(bn)
        # Ziel: Kaufsignale möglichst nah am echten Tief, möglichst wenige; Verkauf halb gewichtet
        obj = (np.mean(bn) if nb else 0) - 0.05 * nb / max(lows, 1) + 0.5 * (np.mean(sn) if sn else 0)
        res[part] = {"n": nb, "near": float(np.mean(bn) * 100) if nb else 0,
                     "hit": float(np.mean(np.array(rets) > 0) * 100) if rets else 0,
                     "per_low": round(nb / max(lows, 1), 2), "obj": float(obj)}
    return S, res


def calibrate(items, mode):
    t0 = time.time()
    M = MODES[mode]
    d_cfg = M["default"]
    S0, base_res = evaluate(items, d_cfg, mode)
    best = (d_cfg, S0, base_res)
    G = M["grid"]
    for wm in G["wm"]:
        for gb in G["gb"]:
            for kh in G["kh"]:
                for km in G["km"]:
                    for sh in G["share"]:
                        cfg = {"wm": wm, "gb": gb, "kh": kh, "km": km, "share": sh}
                        S, r = evaluate(items, cfg, mode)
                        if r["train"]["n"] >= 15 and r["train"]["obj"] > best[2]["train"]["obj"]:
                            best = (cfg, S, r)
    cfg, S, r = best
    use = r["test"]["n"] >= 8 and r["test"]["obj"] >= base_res["test"]["obj"]
    ch = (cfg, S, r) if use else (d_cfg, S0, base_res)
    log(f"Kalibrierung {mode} ({time.time() - t0:.0f}s): Test neu {r['test']} / Standard {base_res['test']} "
        f"-> {'neu' if use else 'Standard'}")
    return {"cfg": ch[0], "S": ch[1], "res": ch[2], "default": base_res, "adopted": bool(use)}


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
        if c.get("v") == CALIB_VERSION and age < CALIB_MAX_AGE_DAYS * 86400:
            log(f"Kalibrierung von {c['date']} wird weiterverwendet.")
            return c
    except Exception as e:
        log("Keine passende frühere Kalibrierung:", e)
    return None


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
    hh300, ll300 = h.rolling(300, min_periods=120).max(), l.rolling(300, min_periods=120).min()
    rpos3 = (c - ll300) / (hh300 - ll300).replace(0, np.nan)
    cyc, cyc_per = cycles(c.values)
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
        "Lage in 300-Kerzen-Spanne": (lin(rpos3, 0, 1), rpos3, lambda x: f"{x * 100:.0f} %",
                                      lambda x: "nahe Langzeit-Tief" if x < 0.15 else "nahe Langzeit-Hoch" if x > 0.85 else "Mitte"),
        "Zyklen": (pd.Series(50 - 50 * cyc), pd.Series(cyc), lambda x: f"{x:+.2f}",
                   lambda x: "Zyklus nahe Tief" if x < -0.6 else "Zyklus nahe Hoch" if x > 0.6 else
                   "Zyklus steigt" if x < 0 else "Zyklus fällt"),
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
    for name, (avail, sc, raw, note, vf, disp) in macro.items():
        ind[name] = (pd.Series(align(ct, avail, sc)), pd.Series(align(ct, avail, raw)), vf, note,
                     pd.Series(align(ct, avail, disp)))
    if dom is not None:
        z = dom
        ind["BTC-Dominanz"] = (pd.Series(50 - 50 * np.tanh(z / 1.5)), pd.Series(z), lambda x: f"{x:+.1f} σ",
                               lambda x: "steigt (Geld fließt zu BTC)" if x > 0.5 else "fällt (gut für Altcoins)"
                               if x < -0.5 else "stabil")

    subs = pd.DataFrame({name: np.asarray(v[0], dtype=float) for name, v in ind.items()})
    return {"subs": subs, "ind": ind, "ct": ct, "t": df["t"].values, "c": df["c"].values,
            "h": df["h"].values, "l": df["l"].values, "cyc_per": cyc_per}


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


def r3(x):
    return None if x is None or not np.isfinite(x) else float(f"{x:.3g}")


def mode_stats(c, buys, sells, lo_idx, N):
    H, n, res = STAT_HORIZON, len(c), {}
    rets = [c[i + H] / c[i] - 1 for i in buys if i + H < n]
    if rets:
        d = [int(np.min(np.abs(lo_idx - i))) for i in buys if len(lo_idx) and i < n - N]
        res["buy"] = [len(buys), round(100 * float(np.mean(rets)), 2), round(100 * float(np.mean(np.array(rets) > 0))),
                      round(float(np.mean(d)), 1) if d else None]
    for k, s in enumerate(sells, 1):
        rr = [c[i + H] / c[i] - 1 for i in s if i + H < n]
        if rr:
            res[f"s{k}"] = [len(s), round(100 * float(np.mean(rr)), 2), round(100 * float(np.mean(np.array(rr) < 0)))]
    return res


def tf_output(df, per_mode, details, extra):
    out = {"c": [[int(r.t), sig(r.o), sig(r.h), sig(r.l), sig(r.c)] for r in df.itertuples()]}
    n = len(df)
    for mode, M in MODES.items():
        sfx = "" if mode == "short" else "l"
        N = M["pivN"] if n >= 3 * M["pivN"] else max(3, n // 8)
        hi, lo = pivot_idx(df["h"].values, df["l"].values, N)
        out["p" + sfx] = [[int(df["t"].iloc[i]), "H"] for i in hi] + [[int(df["t"].iloc[i]), "T"] for i in lo]
        pm = per_mode.get(mode)
        if pm is None:
            continue
        score, groups = pm
        out["z" + sfx] = [None if np.isnan(x) else round(float(x), 1) for x in score]
        buys, sells = signals(score)
        out["sg" + sfx] = [[i, "B"] for i in buys] + [[i, f"S{k}"] for k, s in enumerate(sells, 1) for i in s]
        out["st" + sfx] = mode_stats(df["c"].values, buys, sells, lo, N)
        out["k" + sfx] = groups
    if details:
        out["b"] = details
    out.update(extra)
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
    os.makedirs(os.path.join(OUT_DIR, "data", "ind"), exist_ok=True)

    # 1) Kerzen laden
    loaded, seen = [], set()
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

    # 3) Indikatoren aller Coins
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

    # 4) Kalibrierung je Modus (wöchentlich)
    calib = previous_calibration()
    if calib is None:
        calib = {"v": CALIB_VERSION, "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "modes": {}}
        for mode, M in MODES.items():
            items = []
            gi = {tf: with_groups(C, M["weights"]) for tf, C in idx_C.items()}
            for cc in comps[:CALIB_COINS]:
                g = {tf: with_groups(C, M["weights"]) for tf, C in cc.items() if tf in ("4h", "1d", "1w")}
                for tf in ("4h", "1d"):
                    if tf not in g:
                        continue
                    C = g[tf]
                    H = g.get(HTF[tf])
                    Mi = gi.get(tf)
                    n = len(C["c"])
                    train = np.zeros(n, bool)
                    train[:int(n * 0.7)] = True
                    hi, lo = pivot_idx(C["h"], C["l"], M["pivN"])
                    items.append({"C": C, "H": H, "hidx": htf_map(C, H) if H else None,
                                  "M": Mi, "midx": exact_index(Mi["t"], C["t"]) if Mi else None,
                                  "train": train, "test": ~train, "lo": {M["pivN"]: lo}, "hi": {M["pivN"]: hi}})
            if items:
                calib["modes"][mode] = calibrate(items, mode)
            else:
                calib["modes"][mode] = {"cfg": M["default"], "S": 2.0, "res": None, "default": None, "adopted": False}
    log(f"Kalibrierung fertig nach {time.time() - started:.0f}s")

    # 5) Endgültige Scores + Ausgabe
    mode_idx = {}
    for mode, M in MODES.items():
        cfg = calib["modes"][mode]["cfg"]
        mode_idx[mode] = {tf: combine(with_groups(C, M["weights"]), cfg)[0] for tf, C in idx_C.items()}
    listing = []
    for (meta, src, frames), cc in zip(loaded, comps):
        tfs = {}
        scores_by_mode = {m: {} for m in MODES}
        gm = {mode: {tf: with_groups(C, M["weights"]) for tf, C in cc.items()} for mode, M in MODES.items()}
        bases = {mode: {tf: combine(C, calib["modes"][mode]["cfg"])[0] for tf, C in gm[mode].items()} for mode in MODES}
        for tf, _ in TIMEFRAMES:
            df = frames.get(tf)
            if df is None:
                continue
            C = cc.get(tf)
            per_mode, details, extra = {}, None, {}
            if C is not None:
                for mode in MODES:
                    cal = calib["modes"][mode]
                    cfg, S = cal["cfg"], cal["S"]
                    Cg = gm[mode][tf]
                    hb = mb = None
                    htf = HTF.get(tf)
                    if htf in cc:
                        hb = take(bases[mode][htf], htf_map(C, cc[htf]))
                    if tf in idx_C:
                        mb = take(mode_idx[mode][tf], exact_index(idx_C[tf]["t"], C["t"]))
                    score = to_score(combine(Cg, cfg, hb, mb)[1], S)
                    last = len(score) - 1
                    groups = {g: (None if np.isnan(Cg[k][last]) else round(float(Cg[k][last])))
                              for g, k in (("Lage", "lage"), ("Wende", "wende"), ("Markt", "markt"))}
                    per_mode[mode] = (score, groups)
                    if mode == "short":
                        hv = to_score(hb, S)[-1] if hb is not None else np.nan
                        mv = to_score(mb, S)[-1] if mb is not None else np.nan
                        extra["f"] = {"htf": [htf, None if np.isnan(hv) else round(float(hv))] if htf in cc else None,
                                      "mkt": None if np.isnan(mv) else round(float(mv))}
                last = len(df) - 1
                details = []
                for name, v in C["ind"].items():
                    sc, raw, vf, note = v[0], v[1], v[2], v[3]
                    sv = float(np.asarray(sc, dtype=float)[last])
                    rv = float(np.asarray(raw, dtype=float)[last])
                    if np.isnan(sv):
                        continue
                    details.append([name, GROUP_OF.get(name, "Markt"), round(sv),
                                    "–" if np.isnan(rv) else vf(rv), "–" if np.isnan(rv) else note(rv)])
                extra["cyp"] = C["cyc_per"]
                # Verlauf aller Indikatoren für die Zusatz-Fenster (wird nur bei Bedarf geladen)
                pane = {}
                for name, v in C["ind"].items():
                    series = v[4] if len(v) > 4 else v[1]
                    pane[name] = [r3(x) for x in np.asarray(series, dtype=float)]
                for g, k in (("Score-Teil Lage", "lage"), ("Score-Teil Wende", "wende"), ("Score-Teil Markt", "markt")):
                    pane[g] = [r3(x) for x in gm["short"][tf][k]]
                with open(os.path.join(OUT_DIR, "data", "ind", f"{safe_name(meta['sym'])}_{tf}.json"), "w") as f:
                    json.dump({"t0": int(df["t"].iloc[0]), "v": pane}, f, separators=(",", ":"))
                extra["ind"] = 1
            tfs[tf] = tf_output(df, per_mode, details, extra)
            for mode in MODES:
                key = "z" if mode == "short" else "zl"
                scores_by_mode[mode][tf] = last_valid(tfs[tf].get(key))
        fname = safe_name(meta["sym"])
        with open(os.path.join(OUT_DIR, "data", fname + ".json"), "w", encoding="utf-8") as f:
            json.dump({"s": meta["sym"], "src": src, "tf": tfs}, f, separators=(",", ":"))
        listing.append({"s": meta["sym"], "n": meta["name"], "r": meta["rank"], "f": fname, "src": src,
                        "p": sig(meta["price"] or frames["4h"]["c"].iloc[-1]),
                        "ch": None if meta["chg"] is None else round(float(meta["chg"]), 2),
                        "sc": {tf: scores_by_mode["short"].get(tf) for tf in LIST_TFS},
                        "scl": {tf: scores_by_mode["long"].get(tf) for tf in LIST_TFS}})
    listing.sort(key=lambda c: c["r"])

    mk = {}
    if "fng" in macro:
        mk["fng"] = [float(macro["fng"][0].iloc[-1]), macro.get("fng_label", "")]
    if "dxy" in macro:
        mk["dxy"] = round(float(macro["dxy"][0].iloc[-1]), 2)
    if "global" in macro:
        mk.update(macro["global"])
    payload = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "buy": BUY_LEVEL, "sells": SELL_LEVELS, "reset": RESET_LEVEL, "h": STAT_HORIZON, "tfs": TIMEFRAMES,
               "ltfs": LIST_TFS, "info": IND_INFO, "go": GROUP_OF,
               "modes": {m: {"label": M["label"], "pivN": M["pivN"], "w": M["weights"]} for m, M in MODES.items()},
               "cal": calib, "mkt": mk, "coins": listing}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=float).replace("</", "<\\/")
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html.replace("__DATA__", data))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"updated": payload["updated"]}, f)
    with open(os.path.join(OUT_DIR, "calib.json"), "w") as f:
        json.dump(calib, f, default=float)
    log(f"Fertig: {len(listing)} Coins in {time.time() - started:.0f}s")


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
.dh{position:sticky;top:0;z-index:20;background:rgba(11,14,20,.96);backdrop-filter:blur(10px);
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
.axt{position:absolute;left:0;right:0;bottom:0;height:80px;z-index:6;touch-action:none;
background:linear-gradient(transparent,rgba(96,165,250,.07));border-top:1px solid rgba(96,165,250,.25)}
.axt span{position:absolute;right:8px;bottom:2px;font-size:9.5px;color:#60a5fa99;pointer-events:none}
.layer{position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:3}
.touch{position:absolute;inset:0;z-index:5;touch-action:none}
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
.seg{display:inline-flex;background:var(--card2);border:1px solid var(--line);border-radius:999px;padding:2px;flex:none}
.seg button{border:0;background:none;color:var(--mut);font-size:12px;font-weight:700;padding:5px 10px;border-radius:999px}
.seg button.on{background:var(--acc);color:#06121f}
.plab{position:absolute;left:8px;top:4px;z-index:4;font-size:11px;color:var(--mut);pointer-events:none;
background:#131821cc;padding:1px 6px;border-radius:6px;font-variant-numeric:tabular-nums}
.plab b{color:var(--txt)}
.addb{border:1px dashed var(--line);background:none;color:var(--acc);border-radius:10px;padding:6px 10px;font-size:12.5px;font-weight:600}
#sheet{position:fixed;inset:0;z-index:40;background:#000a;display:none;align-items:flex-end;justify-content:center}
#sheet.open{display:flex}
#sheet .in{background:var(--card);width:100%;max-width:560px;max-height:82vh;overflow-y:auto;border-radius:18px 18px 0 0;
padding:14px 14px calc(14px + env(safe-area-inset-bottom))}
#sheet h4{margin:12px 0 6px;font-size:13px;color:var(--mut)}
#sheet label{display:flex;align-items:center;gap:10px;padding:9px 6px;border-top:1px solid var(--line);font-size:14px}
#sheet input{width:20px;height:20px;accent-color:var(--acc)}
.shead{display:flex;justify-content:space-between;align-items:center;position:sticky;top:-14px;background:var(--card);padding:4px 0 8px}
.grid4{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px}
#dmain{min-width:0}
@media (orientation:landscape) and (max-height:560px){
  #detail.open{display:flex}
  #detail .dh{width:128px;flex:none;height:100vh;overflow-y:auto;border-bottom:0;border-right:1px solid var(--line);
    padding:calc(6px + env(safe-area-inset-top)) 6px 6px calc(6px + env(safe-area-inset-left))}
  #detail .dh .top{flex-direction:column;align-items:stretch;gap:6px}
  #detail .dh .t div{display:none}
  #detail .dh .t b{font-size:14px}
  #detail .dh .back{width:100%;height:32px}
  #detail .dh .sc{font-size:15px;padding:3px 0}
  #detail .dh .fresh{justify-content:center;padding:5px 6px}
  #tfc{flex-wrap:wrap;overflow:visible;gap:4px;margin-top:6px}
  #tfc .chip{padding:5px 8px;font-size:12px;flex:1 0 40%}
  #dmain{flex:1;height:100vh;overflow-y:auto}
  #dmain .box{margin:0;border-radius:0;padding:4px 4px 0}
  #dmain .box h3{margin:0 4px 2px}
  #dmain .mres{position:absolute;left:150px;right:260px;top:12px;margin:0;text-align:center;z-index:2}
  #dmain .box h3{min-height:40px}
  #dmain .sec,#dmain .note{display:none}
}
</style>
</head>
<body>
<header>
  <div class="hrow"><h1>Krypto Score</h1><button class="fresh" id="ref"></button></div>
  <div class="mbar" id="mbar"></div>
  <input class="search" id="q" type="search" placeholder="Coin suchen (z. B. BTC, Solana)…" autocomplete="off">
  <div class="chips"><span class="lbl">Modus:</span><div class="seg mseg"><button data-m="short">Kurzfristig</button><button data-m="long">Langfristig</button></div></div>
  <div class="chips" id="sort"><span class="lbl">Sortieren:</span>
    <button class="chip" data-o="mc">Marktkap.</button>
    <button class="chip" data-o="hi">Score hoch → tief</button>
    <button class="chip" data-o="lo">Score tief → hoch</button>
  </div>
  <div class="chips" id="ltf"><span class="lbl">Score:</span></div>
  <div class="chips" id="chips">
    <button class="chip on" data-f="all">Alle</button>
    <button class="chip b" data-f="buy">Kaufzone ≥80</button>
    <button class="chip" data-f="pos">Kauf 60–80</button>
    <button class="chip" data-f="neu">Neutral</button>
    <button class="chip" data-f="neg">Verkauf 30–40</button>
    <button class="chip s" data-f="sell">Verkaufszone ≤30</button>
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
  <div id="dmain">
  <div class="box"><h3><div class="seg mseg"><button data-m="short">Kurz</button><button data-m="long">Lang</button></div>
    <span style="display:flex;gap:6px"><button class="addb" id="addp">+ Indikator</button><button class="chip sm" id="now">Jetzt ⇥</button></span></h3>
    <div class="mres" id="mres"></div>
    <div class="cwrap" id="cwrap"></div>
    <div class="msg" id="noz" style="display:none">Für diesen Zeitrahmen gibt es zu wenige Kerzen für einen Score.</div>
  </div>
  <div class="sec" id="statbox"><h4>Bilanz der Signale (gesamte Historie)</h4><div class="grid2" id="stat"></div><div class="grid4" id="stat2"></div></div>
  <div class="sec" id="fbox"><h4>Filter &amp; Zyklen</h4><div class="grid2" id="filt"></div></div>
  <div class="sec" id="catbox"><h4>Bestandteile</h4><div class="grid3" id="cats"></div></div>
  <div class="sec" id="pcbox"><div class="grid2">
    <div><h4 class="up">Spricht für Kaufen</h4><div id="pro"></div></div>
    <div><h4 class="dn">Spricht für Verkaufen</h4><div id="con"></div></div>
  </div></div>
  <div class="sec" id="allbox"><details><summary>Alle Indikatoren</summary><table id="all"></table></details></div>
  <div class="sec exp"><details><summary>Wie entsteht der Score?</summary><div class="expc"></div></details></div>
  <p class="note">▲ Kauf = erster Eintritt in die Kaufzone je Runde. ▼ ⅓ / ⅔ / 3/3 = Verkauf in Tranchen bei Score 30 / 20 / 10.
  Der Score nutzt nur Daten bis zur jeweiligen Kerze. Senkrechte gestrichelte Linien = echte Hochs (rot) und Tiefs (grün),
  die erst im Nachhinein feststehen. Datenquelle: <span id="src"></span>. Keine Anlageberatung.</p>
  </div>
</div>

<div id="sheet"><div class="in">
  <div class="shead"><b>Indikatoren unter dem Chart</b><button class="chip sm" id="sheetok">Fertig</button></div>
  <div id="sheetl"></div>
</div></div>

<script>window.DATA = __DATA__;</script>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
 onerror="var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js';document.head.appendChild(s)"></script>
<script>
(function(){
const D = window.DATA, BUY = D.buy, SELLS = D.sells, SELL = SELLS[0];
const TFL = Object.fromEntries(D.tfs);
const SEC = {'1h':3600,'4h':14400,'1d':86400,'5d':432000,'1w':604800,'2w':1209600,'1M':2629800,'6M':15778800,'1y':31557600};
const $ = id => document.getElementById(id);
const store = (k, v) => { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } };
let filter = 'all', query = '', ltf = store('ltf') || '4h', ctf = store('ctf') || '4h', sortBy = store('sort') || 'mc';
let mode = store('mode') === 'long' ? 'long' : 'short';
let panes = []; try { panes = JSON.parse(store('panes') || '[]') || []; } catch (e) { panes = []; }
let cur = null, G = null, view = null, liveTs = 0, liveMap = null, reqId = 0, loadingOld = false;
const cache = {}, indCache = {};
const AXH = () => landscape() ? 56 : 80;   // Höhe der Zeitachse (Schieben/Zoomen)
const sfx = () => mode === 'long' ? 'l' : '';

const color = s => s == null ? '#4b5563' : s >= BUY ? '#22c55e' : s >= 60 ? '#84cc16' : s > 40 ? '#eab308' : s > SELL ? '#f97316' : '#ef4444';
const label = s => s == null ? 'kein Score' : s >= BUY ? 'Kaufzone' : s >= 60 ? 'Kauf' : s > 40 ? 'Neutral' : s > SELL ? 'Verkauf' : 'Verkaufszone';
function price(p){
  if (p == null || !isFinite(p)) return '–';
  const a = Math.abs(p), d = a >= 1000 ? 0 : a >= 1 ? 2 : a >= 0.01 ? 4 : 8;
  return '$' + p.toLocaleString('de-DE', {minimumFractionDigits: d, maximumFractionDigits: d});
}
const pct = x => x == null || !isFinite(x) ? '' : (x >= 0 ? '+' : '') + x.toFixed(2).replace('.', ',') + ' %';
const num = x => x == null || !isFinite(x) ? '–' : Math.abs(x) >= 1000 ? Math.round(x).toLocaleString('de-DE') : String(+x.toPrecision(3)).replace('.', ',');
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
const landscape = () => window.matchMedia('(orientation: landscape) and (max-height: 560px)').matches;

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

// ---------------- Modus
function setMode(m){
  mode = m; store('mode', m);
  document.querySelectorAll('.mseg button').forEach(b => b.classList.toggle('on', b.dataset.m === m));
  render(); explain();
  if (cur && cur.data) show(true);
}
document.querySelectorAll('.mseg').forEach(s => s.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setMode(b.dataset.m); }));

// ---------------- Liste
$('ltf').insertAdjacentHTML('beforeend', D.ltfs.map(t => `<button class="chip" data-t="${t}">${TFL[t]}</button>`).join(''));
function setLtf(t){ ltf = t; store('ltf', t); document.querySelectorAll('#ltf .chip').forEach(b => b.classList.toggle('on', b.dataset.t === t)); render(); }
function setSort(o){ sortBy = o; store('sort', o); document.querySelectorAll('#sort .chip').forEach(b => b.classList.toggle('on', b.dataset.o === o)); render(); }
$('ltf').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setLtf(b.dataset.t); });
$('sort').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setSort(b.dataset.o); });
const scoreOf = c => (mode === 'long' ? c.scl : c.sc)[ltf];
function match(c){
  if (query && !(c.s.toLowerCase().includes(query) || c.n.toLowerCase().includes(query))) return false;
  const s = scoreOf(c);
  if (filter !== 'all' && s == null) return false;
  switch (filter){
    case 'buy': return s >= BUY; case 'pos': return s >= 60 && s < BUY; case 'neu': return s > 40 && s < 60;
    case 'neg': return s > SELL && s <= 40; case 'sell': return s <= SELL;
  }
  return true;
}
function render(){
  const all = D.coins.map(scoreOf).filter(s => s != null);
  const avg = all.length ? all.reduce((a, b) => a + b, 0) / all.length : null;
  $('stats').innerHTML =
    `<div class="stat"><b style="color:${color(avg)}">${avg == null ? '–' : avg.toFixed(0)}</b><span>Ø Score (${TFL[ltf]})</span></div>` +
    `<div class="stat"><b class="up">${all.filter(s => s >= BUY).length}</b><span>in Kaufzone</span></div>` +
    `<div class="stat"><b class="dn">${all.filter(s => s <= SELL).length}</b><span>in Verkaufszone</span></div>`;
  const rows = D.coins.filter(match);
  if (sortBy === 'mc') rows.sort((a, b) => a.r - b.r);
  else rows.sort((a, b) => { const x = scoreOf(a), y = scoreOf(b); if (x == null) return 1; if (y == null) return -1; return sortBy === 'hi' ? y - x : x - y; });
  $('list').innerHTML = rows.length ? rows.map(c => {
    const s = scoreOf(c), col = color(s);
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

// ---------------- Erklärung
function explain(){
  const MD = D.modes[mode], cal = D.cal.modes[mode], cfg = cal.cfg, r = cal.res, d0 = cal.default;
  const names = Object.keys(D.info), grp = g => names.filter(k => D.go[k] === g);
  const li = k => `<li><b>${esc(k)}</b> (Gewicht ${String(MD.w[k] ?? 1).replace('.', ',')}): ${esc(D.info[k])}</li>`;
  const p = x => Math.round(x * 100) + ' %';
  const calTxt = r ? `<p><b>Auto-Kalibrierung</b> (${new Date(D.cal.date).toLocaleDateString('de-DE')}, wöchentlich):
    Die Einstellungen wurden an der Historie der Top-40-Coins (4 Std + 1 T) so gewählt, dass Kaufsignale <b>möglichst selten
    und möglichst nah am echten Tief</b> liegen. Eingestellt an den ersten 70 % der Daten, kontrolliert an den letzten 30 %.
    ${cal.adopted ? 'Die neue Einstellung war auch im Kontrollzeitraum besser.' : 'Im Kontrollzeitraum war keine Einstellung besser – es gilt der Standard.'}
    Kontrollzeitraum: <b>${r.test.n} Kaufsignale</b>, Nähe zum Tief <b>${Math.round(r.test.near)} %</b>
    (100 % = genau am Tief, 0 % = ${MD.pivN}+ Kerzen daneben), Gewinn nach ${D.h} Kerzen in <b>${Math.round(r.test.hit)} %</b> der Fälle${d0 ? ` (Standard: ${d0.test.n} Signale, Nähe ${Math.round(d0.test.near)} %)` : ''}.</p>` : '';
  const html = `
  <p><b>Modus ${esc(MD.label)}</b>: ${mode === 'long' ? 'sucht nur die großen Zyklus-Tiefs und -Hochs (echtes Hoch/Tief = Extrem über ±30 Kerzen), mit mehr Gewicht auf langfristigen Indikatoren und dem übergeordneten Zeitrahmen. Signale sind sehr selten.' : 'sucht die kleineren Tiefs und Hochs (Extrem über ±10 Kerzen) für kürzere Trades.'}</p>
  <p>Jeder Indikator wird für jede Kerze in einen <b>Teil-Score von 0 bis 100</b> umgerechnet (100 = Kaufen, 0 = Verkaufen):</p>
  <p><b>1. Lage</b> – Wie weit unten oder oben steht der Kurs?</p><ul>${grp('Lage').map(li).join('')}</ul>
  <p><b>2. Markt</b> (${p(cfg.wm / (1 + cfg.wm))} der Richtung) – Stimmung und Umfeld:</p><ul>${grp('Markt').map(li).join('')}</ul>
  <p><b>3. Wende</b> – Gibt es Zeichen, dass der Kurs dreht?</p><ul>${grp('Wende').map(li).join('')}</ul>
  <p><b>Verbindung:</b> Lage und Markt bestimmen die <b>Richtung</b>, die Wende die <b>Bestätigung</b>. Fällt der Kurs noch
  („fallendes Messer“), wird das Signal auf ${p(Math.max(0.05, 2 * cfg.gb - 1))}–${p(cfg.gb)} gedämpft.
  Danach wird es mit dem <b>übergeordneten Zeitrahmen</b> (Stärke ${String(cfg.kh).replace('.', ',')}) und dem <b>Gesamtmarkt</b>
  (Top-20-Index, Stärke ${String(cfg.km).replace('.', ',')}) abgeglichen. Die Spreizung (Faktor ${cal.S.toFixed(2).replace('.', ',')})
  ist so gewählt, dass rund ${String(cfg.share * 100).replace('.', ',')} % der Kerzen in einer Zone liegen.</p>
  <p><b>Signale:</b> ▲ <b>Kauf</b> beim ersten Eintritt in ≥ ${BUY}. ▼ <b>Verkauf in drei Tranchen</b>: ⅓ bei ≤ ${SELLS[0]},
  ⅔ bei ≤ ${SELLS[1]}, Rest bei ≤ ${SELLS[2]}. Eine neue Runde beginnt erst, wenn der Score wieder ${D.reset} erreicht – so gibt
  es pro Tief nur ein Kaufsignal.</p>${calTxt}
  <p><b>Ehrlichkeit:</b> Jede Kerze nutzt nur Daten bis zu dieser Kerze – alte Signale ändern sich nicht. Die gestrichelten
  Hoch/Tief-Linien stehen erst im Nachhinein fest und zeigen nur, wie gut der Score getroffen hätte. Kein Indikator kann
  Tiefs sicher vorhersagen. Keine Anlageberatung.</p>`;
  document.querySelectorAll('.expc').forEach(el => el.innerHTML = html);
}

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
    if (cur && cur.data) await liveChart();
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
    else if (t > lastT) { T.c.push(row); if (T.z) T.z.push(null); if (T.zl) T.zl.push(null); }
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
  const T = cur && cur.data && cur.data.tf[ctf];
  if (!T || loadingOld || T.noMore || !['1h', '4h'].includes(ctf)) return;
  loadingOld = true; $('mres').innerHTML = '<span style="color:var(--acc)">Lade ältere Kerzen…</span>';
  try {
    const first = T.c[0][0];
    const k = await bget(`/api/v3/klines?symbol=${cur.meta.s}USDT&interval=${ctf}&endTime=${first * 1000 - 1}&limit=1000`);
    const rows = k.map(r => [Math.floor(r[0] / 1000), +r[1], +r[2], +r[3], +r[4]]).filter(r => r[0] < first);
    if (!rows.length) T.noMore = true;
    else {
      const pad = new Array(rows.length).fill(null);
      T.c = rows.concat(T.c); if (T.z) T.z = pad.concat(T.z); if (T.zl) T.zl = pad.concat(T.zl);
      T.pre = (T.pre || 0) + rows.length; view.to += rows.length; show(true);
    }
    $('mres').innerHTML = rows.length ? `${rows.length} ältere Kerzen geladen (ohne Score)` : 'Anfang der Historie erreicht.';
  } catch (e) { T.noMore = true; $('mres').textContent = 'Ältere Kerzen nicht verfügbar.'; }
  loadingOld = false;
}

// ---------------- Indikator-Auswahl
const REF = {'RSI (14)': [30, 70], 'Stochastic RSI': [20, 80], 'Williams %R': [-80, -20], 'CCI (20)': [-100, 100],
  'MFI (14)': [20, 80], 'Bollinger %B': [0, 1], 'Fear & Greed': [25, 75], 'Zyklen': [-0.6, 0.6],
  'Score-Teil Lage': [40, 60], 'Score-Teil Wende': [40, 60], 'Score-Teil Markt': [40, 60],
  'Lage in 100-Kerzen-Spanne': [0.15, 0.85], 'Lage in 300-Kerzen-Spanne': [0.15, 0.85]};
function sheetList(){
  const names = Object.keys(D.info), grp = g => names.filter(k => D.go[k] === g);
  const box = k => `<label><input type="checkbox" value="${esc(k)}" ${panes.includes(k) ? 'checked' : ''}> ${esc(k)}</label>`;
  $('sheetl').innerHTML = `<p class="msg" style="text-align:left">Bis zu 4 Fenster. Jedes läuft synchron mit Chart, Zeitachse und Hoch/Tief-Linien.</p>` +
    `<h4>Score-Bestandteile</h4>${['Score-Teil Lage', 'Score-Teil Wende', 'Score-Teil Markt'].map(box).join('')}` +
    `<h4>Zyklen</h4>${box('Zyklen')}` +
    ['Lage', 'Wende', 'Markt'].map(g => `<h4>${g}</h4>` + grp(g).filter(k => k !== 'Zyklen').map(box).join('')).join('');
}
$('addp').addEventListener('click', () => { sheetList(); $('sheet').classList.add('open'); });
$('sheetl').addEventListener('change', e => {
  const k = e.target.value;
  if (e.target.checked) { if (panes.length >= 4) { e.target.checked = false; return; } panes.push(k); }
  else panes = panes.filter(x => x !== k);
  store('panes', JSON.stringify(panes));
});
const closeSheet = () => { $('sheet').classList.remove('open'); if (cur && cur.data) show(true); };
$('sheetok').addEventListener('click', closeSheet);
$('sheet').addEventListener('click', e => { if (e.target.id === 'sheet') closeSheet(); });

// ---------------- Detail
function destroy(){ if (G) G.charts.forEach(c => c.chart.remove()); G = null; }
function closeCoin(){ destroy(); cur = null; $('detail').classList.remove('open'); document.body.style.overflow = ''; }
$('tfc').innerHTML = D.tfs.map(([k, l]) => `<button class="chip" data-t="${k}">${l}</button>`).join('');
$('tfc').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b && !b.disabled) { ctf = b.dataset.t; store('ctf', ctf); view = null; M = null; show(); } });
function header(){
  const meta = cur.meta;
  $('dt').textContent = landscape() ? meta.s : meta.s + ' · ' + meta.n;
  $('ds').innerHTML = price(meta.p) + ' <span class="' + (meta.ch >= 0 ? 'up' : 'dn') + '">' + pct(meta.ch) + '</span> · #' + meta.r;
}
async function openCoin(sym){
  const meta = D.coins.find(x => x.s === sym); if (!meta) return closeCoin();
  destroy(); view = null; M = null;
  $('detail').classList.add('open'); $('dmain').scrollTop = 0; $('detail').scrollTop = 0; document.body.style.overflow = 'hidden';
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
  cur.data = cache[meta.f]; $('mres').textContent = '';
  show();
}
function waitLib(cb, n){ if (window.LightweightCharts) cb(); else if ((n || 0) < 100) setTimeout(() => waitLib(cb, (n || 0) + 1), 100); }
async function loadInd(T){
  if (!panes.length || !T.ind) return null;
  const key = cur.meta.f + '_' + ctf;
  if (!indCache[key]) {
    try { const r = await fetch('data/ind/' + key + '.json?v=' + encodeURIComponent(D.updated)); indCache[key] = r.ok ? await r.json() : {v: {}}; }
    catch (e) { indCache[key] = {v: {}}; }
  }
  return indCache[key];
}
async function show(keep){
  if (!cur || !cur.data) return;
  const tfs = cur.data.tf;
  if (!tfs[ctf]) ctf = tfs['4h'] ? '4h' : Object.keys(tfs)[0];
  document.querySelectorAll('#tfc .chip').forEach(b => { b.disabled = !tfs[b.dataset.t]; b.classList.toggle('on', b.dataset.t === ctf); });
  document.querySelectorAll('.mseg button').forEach(b => b.classList.toggle('on', b.dataset.m === mode));
  header();
  const T = tfs[ctf], Z = T['z' + sfx()], hasZ = !!Z;
  const s = hasZ ? Z.slice().reverse().find(x => x != null) : null, col = color(s);
  $('dsc').textContent = s == null ? '–' : Math.round(s); $('dsc').style.color = col; $('dsc').style.background = col + '22';
  ['fbox', 'catbox', 'pcbox', 'allbox', 'statbox'].forEach(id => $(id).style.display = hasZ && T.b ? '' : 'none');
  $('noz').style.display = hasZ ? 'none' : '';
  if (hasZ && T.b) details(T);
  if (!keep) M = null;
  const ind = await loadInd(T);
  destroy();
  waitLib(() => draw(T, ind));
}

function details(T){
  const st = T['st' + sfx()] || {}, H = D.h, k = T['k' + sfx()] || {};
  const b = st.buy;
  $('stat').innerHTML = (b ? `<div class="cat">▲ Kauf-Signale (${b[0]}×)<b style="color:${b[2] >= 50 ? 'var(--buy)' : 'var(--sell)'}">${b[2]} % Treffer</b>
      Ø ${pct(b[1])} nach ${H} Kerzen</div><div class="cat">Abstand zum echten Tief<b>${b[3] == null ? '–' : String(b[3]).replace('.', ',') + ' Kerzen'}</b>im Schnitt (0 = genau am Tief)</div>`
    : '<div class="cat">▲ Kauf-Signale<b>–</b>keine in der Historie</div><div class="cat">Abstand zum Tief<b>–</b></div>');
  $('stat2').innerHTML = [1, 2, 3].map(i => { const v = st['s' + i], lab = ['⅓', '⅔', '3/3'][i - 1];
    return v ? `<div class="cat">▼ ${lab} (${v[0]}×)<b style="color:${v[2] >= 50 ? 'var(--buy)' : 'var(--sell)'}">${v[2]} %</b>Ø ${pct(v[1])}</div>`
      : `<div class="cat">▼ ${lab}<b>–</b>keine</div>`; }).join('');
  const f = T.f || {};
  const cyp = (T.cyp || []).map(p => p[0]).join(' / ');
  $('filt').innerHTML = (f.htf ? `<div class="cat">Übergeordnet (${TFL[f.htf[0]]})<b style="color:${color(f.htf[1])}">${f.htf[1] ?? '–'}</b>${label(f.htf[1])}</div>` : '<div class="cat">Übergeordnet<b>–</b>nicht vorhanden</div>') +
    `<div class="cat">Gesamtmarkt (Top 20)<b style="color:${color(f.mkt)}">${f.mkt ?? '–'}</b>${label(f.mkt)}</div>` +
    `<div class="cat" style="grid-column:1/3">Stärkste Zyklen (Wellenlänge)<b>${cyp ? cyp + ' Kerzen' : '–'}</b>${cyp ? '≈ ' + (T.cyp || []).map(p => dur(p[0] * SEC[ctf])).join(' / ') : ''}</div>`;
  $('cats').innerHTML = ['Lage', 'Markt', 'Wende'].map(g => {
    const v = k[g]; if (v == null) return `<div class="cat">${g}<b>–</b></div>`;
    return `<div class="cat">${g}<b style="color:${color(v)}">${v}</b><div class="bar"><i style="width:${v}%;background:${color(v)}"></i></div></div>`;
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

// ---------------- Charts: Kurs, Score und gewählte Indikatoren untereinander
function heights(k, hasZ){
  if (landscape()) {
    const box = $('cwrap').getBoundingClientRect(), top = Math.max(0, box.top - $('dmain').getBoundingClientRect().top);
    const total = Math.max(220, window.innerHeight - top - AXH() - 4);
    const zh = hasZ ? Math.max(56, Math.round(total * 0.2)) : 0, ph = k ? Math.max(46, Math.round(total * 0.13)) : 0;
    return {price: Math.max(110, total - zh - ph * k), score: zh, pane: ph};
  }
  return {price: 320, score: hasZ ? 150 : 0, pane: 105};
}
function draw(T, ind){
  const L = window.LightweightCharts, Z = T['z' + sfx()], hasZ = !!Z, pre = T.pre || 0;
  const wrap = $('cwrap');
  const list = [{kind: 'price'}];
  if (hasZ) list.push({kind: 'score'});
  if (ind && ind.v) panes.forEach(name => { if (ind.v[name]) list.push({kind: 'ind', name}); });
  const H = heights(list.length - 1 - (hasZ ? 1 : 0), hasZ);
  wrap.innerHTML = list.map((p, i) => {
    const h = (p.kind === 'price' ? H.price : p.kind === 'score' ? H.score : H.pane) + (i === list.length - 1 ? AXH() : 0);
    const lab = p.kind === 'ind' ? `<div class="plab" id="plab${i}">${esc(p.name)}</div>` : p.kind === 'score' ? `<div class="plab" id="plab${i}">Score · ${esc(D.modes[mode].label)}</div>` : '';
    return `<div class="cw" style="height:${h}px"><div class="ch" id="ch${i}" style="height:${h}px"></div>${lab}
      ${p.kind === 'price' ? '<div class="layer" id="ml"></div>' : ''}
      <div class="touch" data-i="${i}" style="bottom:${i === list.length - 1 ? AXH() : 0}px"></div></div>`;
  }).join('') + `<div class="layer" id="pvl" style="bottom:${AXH()}px"></div>
    <div class="axt" id="axt" style="height:${AXH()}px"><span>Zeitachse: ziehen = schieben · 2 Finger = zoomen</span></div>`;
  const opt = last => ({
    autoSize: true, handleScroll: false, handleScale: false,
    layout: {background: {type: 'solid', color: '#131821'}, textColor: '#8b93a5', fontSize: 11},
    grid: {vertLines: {color: '#1c2230'}, horzLines: {color: '#1c2230'}},
    rightPriceScale: {borderColor: '#242b38', minimumWidth: 78},
    timeScale: {visible: last, borderColor: '#242b38', timeVisible: !big(), secondsVisible: false, rightOffset: 0, minBarSpacing: 0.05,
      minimumHeight: AXH(), lockVisibleTimeRangeOnResize: true,
      tickMarkFormatter: (t, type) => { const d = new Date(t * 1000);
        if (type === 0) return String(d.getFullYear());
        if (type === 1) return d.toLocaleDateString('de-DE', {month: 'short'});
        if (type === 2) return d.getDate() + '.' + (d.getMonth() + 1) + '.';
        return d.toLocaleTimeString('de-DE', {hour: '2-digit', minute: '2-digit'}); }},
    crosshair: {mode: L.CrosshairMode.Normal, vertLine: {color: '#60a5fa', labelBackgroundColor: '#2563eb'},
      horzLine: {color: '#60a5fa', labelBackgroundColor: '#2563eb'}},
    localization: {locale: 'de-DE', timeFormatter: t => fmtDate(t)},
  });
  const n = T.c.length, lastP = T.c[n - 1][4];
  const prec = lastP >= 1000 ? 2 : lastP >= 1 ? 3 : lastP >= 0.01 ? 5 : 8;
  const charts = [];
  list.forEach((p, i) => {
    const chart = L.createChart($('ch' + i), opt(i === list.length - 1));
    let series, vals = null;
    if (p.kind === 'price') {
      series = chart.addCandlestickSeries({upColor: '#22c55e', downColor: '#ef4444', borderVisible: false,
        wickUpColor: '#22c55e', wickDownColor: '#ef4444', priceFormat: {type: 'price', precision: prec, minMove: Math.pow(10, -prec)}});
      series.setData(T.c.map(k => ({time: k[0], open: k[1], high: k[2], low: k[3], close: k[4]})));
      const sg = T['sg' + sfx()] || [], mk = [];
      const LBL = {B: ['Kauf', '#22c55e', 'belowBar', 'arrowUp', 1.3], S1: ['⅓', '#fb923c', 'aboveBar', 'arrowDown', 0.8],
        S2: ['⅔', '#f87171', 'aboveBar', 'arrowDown', 0.9], S3: ['3/3', '#ef4444', 'aboveBar', 'arrowDown', 1]};
      sg.forEach(([idx, code]) => { const j = idx + pre; if (j < 0 || j >= n) return; const m = LBL[code];
        mk.push({time: T.c[j][0], position: m[2], color: m[1], shape: m[3], text: m[0], size: m[4]}); });
      mk.sort((a, b) => a.time - b.time);
      series.setMarkers(mk);
    } else if (p.kind === 'score') {
      vals = Z;
      series = chart.addBaselineSeries({baseValue: {type: 'price', price: 50}, lineWidth: 2,
        topLineColor: '#22c55e', topFillColor1: 'rgba(34,197,94,.28)', topFillColor2: 'rgba(34,197,94,.02)',
        bottomLineColor: '#ef4444', bottomFillColor1: 'rgba(239,68,68,.02)', bottomFillColor2: 'rgba(239,68,68,.28)',
        priceFormat: {type: 'price', precision: 0, minMove: 1}, lastValueVisible: true, priceLineVisible: false,
        autoscaleInfoProvider: () => ({priceRange: {minValue: 0, maxValue: 100}})});
      series.setData(T.c.map((k, j) => Z[j] == null ? {time: k[0]} : {time: k[0], value: Z[j]}));
      series.createPriceLine({price: BUY, color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Kauf'});
      SELLS.forEach((lv, q) => series.createPriceLine({price: lv, color: ['#fb923c', '#f87171', '#ef4444'][q], lineWidth: 1, lineStyle: 2, axisLabelVisible: q === 0, title: q === 0 ? 'Verk.' : ''}));
    } else {
      const src = ind.v[p.name];
      vals = T.c.map((k, j) => { const q = j - pre; return q >= 0 && q < src.length ? src[q] : null; });
      series = chart.addLineSeries({color: '#a78bfa', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: true,
        priceFormat: {type: 'custom', formatter: v => num(v), minMove: 0.0001}});
      series.setData(T.c.map((k, j) => vals[j] == null ? {time: k[0]} : {time: k[0], value: vals[j]}));
      (REF[p.name] || []).forEach(v => series.createPriceLine({price: v, color: '#4b5563', lineWidth: 1, lineStyle: 2, axisLabelVisible: false}));
    }
    charts.push({kind: p.kind, name: p.name, chart, series, vals, i});
  });
  G = {T, n, charts, pc: charts[0].chart, cs: charts[0].series};
  if (!view) view = {bars: Math.min(n, big() ? 150 : 120), to: n - 1 + 4};
  bindTouch(); clampView(); applyView();
  // alle Preisachsen gleich breit, damit die Linien exakt übereinander liegen
  requestAnimationFrame(() => {
    if (!G) return;
    try {
      const w = Math.max(78, ...G.charts.map(c => c.chart.priceScale('right').width()));
      G.charts.forEach(c => c.chart.applyOptions({rightPriceScale: {minimumWidth: w}}));
    } catch (e) {}
    redraw(); setTimeout(redraw, 150); setTimeout(redraw, 600);
  });
  G.pc.timeScale().subscribeVisibleLogicalRangeChange(() => requestAnimationFrame(redraw));
  labels(n - 1);
}
function redraw(){ drawPiv(); drawM(); }
function labels(i){
  if (!G) return;
  G.charts.forEach(c => { if (c.kind === 'price') return; const el = $('plab' + c.i); if (!el) return;
    const v = c.vals ? c.vals[Math.max(0, Math.min(G.n - 1, i))] : null;
    el.innerHTML = (c.kind === 'score' ? `Score · ${esc(D.modes[mode].label)}` : esc(c.name)) + ` <b>${c.kind === 'score' ? (v == null ? '–' : Math.round(v)) : num(v)}</b>`; });
}

// ---------------- Ansicht: Schieben und Zoomen auf der Zeitachse
const MINB = 10;
const maxBars = () => Math.max(MINB + 1, G.n + 8);
function plotWidth(){
  let w = 78; try { w = G.pc.priceScale('right').width(); } catch (e) {}
  return Math.max(50, $('cwrap').clientWidth - w);
}
function clampView(){
  view.bars = Math.max(MINB, Math.min(maxBars(), view.bars));
  view.to = Math.min(G.n - 1 + 8, Math.max(Math.min(view.bars - 1, G.n + 7), view.to));
}
function applyView(){
  if (!G) return;
  const r = {from: view.to - view.bars, to: view.to};
  G.charts.forEach(c => c.chart.timeScale().setVisibleLogicalRange(r));
  requestAnimationFrame(redraw);
  if (view.to - view.bars < 15) loadOlder();
}
$('now').addEventListener('click', () => { if (!G) return; view.to = G.n - 1 + 4; clampView(); applyView(); });
function cap(el, e){ try { el.setPointerCapture(e.pointerId); } catch (x) {} }

// ---------------- Hoch/Tief-Linien – dauerhaft, durchgehend durch alle Fenster
function drawPiv(){
  const el = $('pvl'); if (!el) return; el.innerHTML = '';
  const P = G && G.T['p' + sfx()]; if (!P) return;
  const ts = G.pc.timeScale(), lines = [];
  for (const [t, k] of P) {
    const x = ts.timeToCoordinate(t); if (x == null) continue;
    lines.push(`<div class="vl" style="left:${x}px;border-color:${k === 'H' ? 'rgba(239,68,68,.6)' : 'rgba(34,197,94,.6)'}"></div>`);
    if (lines.length > 400) break;
  }
  el.innerHTML = lines.join('');
}

// ---------------- Messen: 1 Finger = Start fest, Ende folgt; 2 Finger = jeder Finger hält einen Punkt
let M = null;
const fingers = new Map();
function timeOf(l){
  if (l >= 0 && l < G.n) return G.T.c[Math.round(l)][0];
  const step = SEC[ctf];
  return l < 0 ? G.T.c[0][0] + l * step : G.T.c[G.n - 1][0] + (l - G.n + 1) * step;
}
function pointAt(e){
  const r = $('ch0').getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const l = G.pc.timeScale().coordinateToLogical(x), p = G.cs.coordinateToPrice(y);
  if (l == null || p == null) return null;
  return {l: Math.round(l), p};
}
function cross(idx, priceVal){
  G.charts.forEach(c => {
    if (idx < 0 || idx >= G.n) { c.chart.clearCrosshairPosition(); return; }
    const t = G.T.c[idx][0];
    if (c.kind === 'price') c.chart.setCrosshairPosition(priceVal ?? G.T.c[idx][4], t, c.series);
    else if (c.vals && c.vals[idx] != null) c.chart.setCrosshairPosition(c.vals[idx], t, c.series);
    else c.chart.clearCrosshairPosition();
  });
  labels(idx);
}
function drawM(){
  const layer = $('ml'); if (!layer) return; layer.innerHTML = '';
  if (!G || !M) return;
  const ts = G.pc.timeScale(), pos = q => ({x: ts.logicalToCoordinate(q.l), y: G.cs.priceToCoordinate(q.p)});
  const B = M.b, inData = B.l >= 0 && B.l < G.n;
  if (inData) cross(B.l, B.p); else G.charts.forEach(c => c.chart.clearCrosshairPosition());
  const pb = pos(B), W = $('cwrap').clientWidth; let html = '', res = '';
  if (M.moved) {
    const pa = pos(M.a), lo = M.a.l <= B.l ? M.a : B, hi = M.a.l <= B.l ? B : M.a;
    const up = hi.p >= lo.p, col = up ? '#22c55e' : '#ef4444';
    const dp = hi.p - lo.p, pc_ = dp / lo.p * 100, bars = hi.l - lo.l, d = dur(bars * SEC[ctf]);
    if (pa.x != null && pb.x != null && pa.y != null && pb.y != null)
      html += `<div class="rect" style="left:${Math.min(pa.x, pb.x)}px;top:${Math.min(pa.y, pb.y)}px;width:${Math.abs(pb.x - pa.x)}px;height:${Math.abs(pb.y - pa.y)}px;border-color:${col};background:${col}22"></div>
        <div class="dot" style="left:${pa.x}px;top:${pa.y}px;background:#60a5fa"></div><div class="dot" style="left:${pb.x}px;top:${pb.y}px;background:${col}"></div>`;
    res = `<b style="color:${col}">${pct(pc_)}</b> (${dp >= 0 ? '+' : '−'}${price(Math.abs(dp))}) · ${bars} Kerzen · ${d}`;
    if (pa.x != null && pb.x != null) {
      const cx = Math.min(Math.max((pa.x + pb.x) / 2, 95), W - 95), top = Math.max(64, Math.min(pa.y ?? 0, pb.y ?? 0) - 14);
      html += `<div class="bub" style="left:${cx}px;top:${top}px"><b style="color:${col}">${pct(pc_)}</b> · ${d}<br>${price(lo.p)} → ${price(hi.p)}<br>${fmtDate(timeOf(lo.l))} → ${fmtDate(timeOf(hi.l))}</div>`;
    }
  } else if (pb.x != null && pb.y != null) {
    const top = Math.max(52, pb.y - 18), left = Math.min(Math.max(pb.x, 70), W - 70);
    html += `<div class="bub" style="left:${left}px;top:${top}px"><b>${price(B.p)}</b><br>${fmtDate(timeOf(B.l))}</div>`;
    res = `<span style="color:var(--acc)">${price(B.p)} · ${fmtDate(timeOf(B.l))}</span>`;
  }
  layer.innerHTML = html; $('mres').innerHTML = res;
}
function bindTouch(){
  document.querySelectorAll('#cwrap .touch').forEach(el => {
    const i = +el.dataset.i;
    if (i === 0) {
      el.addEventListener('pointerdown', e => {
        if (!G) return; e.preventDefault(); cap(el, e);
        const q = pointAt(e); if (!q) return;
        if (fingers.size === 0 || !M) { fingers.clear(); fingers.set(e.pointerId, 'b'); M = {a: q, b: q, moved: false, x0: e.clientX, y0: e.clientY}; }
        else if (fingers.size === 1) { fingers.set(e.pointerId, 'a'); M.a = q; M.moved = true; }
        drawM();
      });
      el.addEventListener('pointermove', e => {
        if (!G) return;
        const q = pointAt(e); if (!q) return;
        const role = fingers.get(e.pointerId);
        if (!role) { if (e.pointerType === 'mouse' && fingers.size === 0) { M = {a: q, b: q, moved: false, hover: true}; drawM(); } return; }
        M[role] = q;
        if (!M.moved && Math.hypot(e.clientX - M.x0, e.clientY - M.y0) > 8) M.moved = true;
        drawM();
      });
      ['pointerup', 'pointercancel'].forEach(ev => el.addEventListener(ev, e => fingers.delete(e.pointerId)));
      el.addEventListener('pointerleave', () => { if (M && M.hover) { M = null; drawM(); $('mres').innerHTML = ''; } });
    } else {
      const mv = e => {
        if (!G) return; const c = G.charts[i]; if (!c) return;
        const r = $('ch' + i).getBoundingClientRect(), l = c.chart.timeScale().coordinateToLogical(e.clientX - r.left);
        if (l == null) return; const j = Math.max(0, Math.min(G.n - 1, Math.round(l)));
        M = null; drawM(); cross(j);
        const v = c.vals ? c.vals[j] : null;
        $('mres').innerHTML = `<span style="color:var(--acc)">${c.kind === 'score' ? 'Score ' + (v == null ? '–' : Math.round(v)) : esc(c.name) + ' ' + num(v)}</span> · ${fmtDate(G.T.c[j][0])}`;
      };
      el.addEventListener('pointerdown', e => { e.preventDefault(); cap(el, e); mv(e); });
      el.addEventListener('pointermove', e => { if (e.buttons || e.pointerType !== 'mouse') mv(e); });
    }
  });
  const ax = $('axt'), axP = new Map(); let axStart = null;
  const begin = () => {
    const pts = [...axP.values()], r = $('cwrap').getBoundingClientRect();
    if (!pts.length) { axStart = null; return; }
    const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length - r.left;
    axStart = {to: view.to, bars: view.bars, mx, d: pts.length > 1 ? Math.abs(pts[0].x - pts[1].x) : 0, n: pts.length};
  };
  ax.addEventListener('pointerdown', e => { if (!G) return; e.preventDefault(); cap(ax, e); axP.set(e.pointerId, {x: e.clientX}); begin(); });
  ax.addEventListener('pointermove', e => {
    if (!G || !axStart || !axP.has(e.pointerId)) return;
    axP.set(e.pointerId, {x: e.clientX});
    const pts = [...axP.values()], r = $('cwrap').getBoundingClientRect(), W = plotWidth();
    const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length - r.left;
    let bars = axStart.bars;
    if (axStart.n > 1 && pts.length > 1) { const d = Math.abs(pts[0].x - pts[1].x); if (axStart.d > 10 && d > 10) bars = axStart.bars * axStart.d / d; }
    const anchor = axStart.to - axStart.bars + axStart.mx / W * axStart.bars;
    bars = Math.max(MINB, Math.min(maxBars(), bars));
    view.bars = bars; view.to = anchor - mx / W * bars + bars; clampView(); applyView();
  });
  ['pointerup', 'pointercancel'].forEach(ev => ax.addEventListener(ev, e => { axP.delete(e.pointerId); begin(); }));
  ax.addEventListener('wheel', e => { if (!G) return; e.preventDefault(); view.bars *= e.deltaY > 0 ? 1.15 : 0.87; clampView(); applyView(); }, {passive: false});
}
let rsT = null;
window.addEventListener('resize', () => { clearTimeout(rsT); rsT = setTimeout(() => { if (cur && cur.data) { header(); show(true); } }, 200); });

document.querySelectorAll('.mseg button').forEach(b => b.classList.toggle('on', b.dataset.m === mode));
setLtf(D.ltfs.includes(ltf) ? ltf : '4h');
explain();
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
