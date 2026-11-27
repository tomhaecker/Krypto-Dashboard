#!/usr/bin/env python3
"""
Krypto-Dashboard v4 – läuft automatisch über GitHub Actions (stündlich).

Ablauf:
1. Top-Coins nach Marktkapitalisierung (CoinGecko, Ersatz CoinPaprika), ohne Stablecoins/Wrapped.
2. Kerzen in 9 Zeitrahmen (Binance, Ersatz OKX/KuCoin). Tag/Woche/Monat mit kompletter Historie,
   verlängert um ältere Daten: Bitcoin von Bitstamp (ab 2011) und Blockchain.com (Tageskurse ab 2010),
   ältere Altcoins von Bitfinex.
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
BUY_LEVEL = 80
SELL_LEVELS = (30, 20, 10)   # Verkauf in drei Tranchen
RESET_LEVEL = 50             # neue Signal-Runde erst, wenn der Score wieder die Mitte erreicht
STAT_HORIZON = 10
INDEX_COINS = 20
CALIB_MAX_AGE_DAYS = 7
CALIB_COINS = 40
CALIB_VERSION = 9
EXTEND_RANK = 60       # für Coins bis zu diesem Rang wird ältere Historie bei Bitfinex gesucht
OUT_DIR = "site"

LIST_TFS = ["1h", "4h", "1d", "1w"]
TIMEFRAMES = [("1h", "1 Std"), ("4h", "4 Std"), ("1d", "1 T"), ("5d", "5 T"), ("1w", "1 W"),
              ("2w", "2 W"), ("1M", "1 M"), ("6M", "6 M"), ("1y", "1 J")]
TF_SEC = {"1h": 3600, "4h": 14400, "1d": 86400, "5d": 432000, "1w": 604800, "2w": 1209600,
          "1M": 2629800, "6M": 15778800, "1y": 31557600}
HTF = {"1h": "4h", "4h": "1d", "1d": "1w", "5d": "2w", "1w": "1M", "2w": "1M"}
NATIVE_LIMIT = {"1h": INTRADAY_KEEP, "4h": INTRADAY_KEEP}
FULL_HISTORY = {"1d", "1w", "1M"}

# Gesamt-Score: Kombination mehrerer Zeitrahmen (je nur die zuletzt abgeschlossene Kerze)
OVERALL = {"short": {"1h": 0.25, "4h": 0.40, "1d": 0.35},
           "long": {"1d": 0.30, "1w": 0.45, "1M": 0.25}}
# Bitcoin-Halvings: Grenzen der klassischen Vier-Jahres-Zyklen
HALVINGS = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20", "2028-04-15"]

GROUP_OF = {
    "RSI (14)": "Lage", "Stochastic RSI": "Lage", "Williams %R": "Lage", "CCI (20)": "Lage", "MFI (14)": "Lage",
    "Bollinger %B": "Lage", "Abstand EMA 50": "Lage", "Abstand SMA 200": "Lage",
    "Lage in 100-Kerzen-Spanne": "Lage", "Lage in 300-Kerzen-Spanne": "Lage",
    "MACD dreht": "Wende", "RSI-Divergenz": "Wende", "Stoch-RSI Kreuzung": "Wende", "RSI-Schwung (3)": "Wende",
    "Erholung vom Tief": "Wende", "Volumen-Klimax": "Wende", "Kerzen-Docht": "Wende", "Zyklen": "Wende",
    "Fear & Greed": "Markt", "Dollar-Index (DXY)": "Markt", "Bitcoin/Gold": "Markt", "BTC-Dominanz": "Markt",
    "Kreuz naht": "Wende",
    "BTC 200-Wochen-Ø": "Zyklus", "BTC Mayer-Multiple": "Zyklus", "BTC Pi-Cycle": "Zyklus", "Halving-Phase": "Zyklus",
}
BASE_WEIGHTS = {"Lage in 100-Kerzen-Spanne": 2.5, "Lage in 300-Kerzen-Spanne": 0.5, "Abstand EMA 50": 1.5,
                "Abstand SMA 200": 1.5, "Bollinger %B": 1.0, "RSI (14)": 0.75, "Stochastic RSI": 0.5,
                "Williams %R": 0.5, "CCI (20)": 0.5, "MFI (14)": 0.75,
                "Erholung vom Tief": 2.0, "Stoch-RSI Kreuzung": 1.0, "RSI-Schwung (3)": 1.0, "MACD dreht": 1.0,
                "RSI-Divergenz": 1.0, "Volumen-Klimax": 0.5, "Kerzen-Docht": 0.5, "Zyklen": 1.5,
                "Fear & Greed": 1.0, "Dollar-Index (DXY)": 1.0, "Bitcoin/Gold": 1.0, "BTC-Dominanz": 1.0,
                "Kreuz naht": 1.0, "BTC 200-Wochen-Ø": 1.0, "BTC Mayer-Multiple": 1.0, "BTC Pi-Cycle": 1.0,
                "Halving-Phase": 1.0}
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
    "Kreuz naht": "Nähert sich ein schneller Durchschnitt einem langsamen und wird ihn bald kreuzen? (EMA 20/SMA 50 und "
                  "SMA 50/200). Bullisches Kreuz (z. B. Golden Cross) in Sicht = Kauf, bärisches (Death Cross) = Verkauf.",
    "BTC 200-Wochen-Ø": "Bitcoin-Kurs im Verhältnis zum 200-Wochen-Durchschnitt. Nahe 1× = historisch günstig (Kauf), "
                        "4× und mehr = historisch teuer (Verkauf).",
    "BTC Mayer-Multiple": "Bitcoin-Kurs geteilt durch den 200-Tage-Durchschnitt. Unter 0,8 = günstig, über 2,4 = überhitzt.",
    "BTC Pi-Cycle": "111-Tage-Durchschnitt im Verhältnis zum doppelten 350-Tage-Durchschnitt. Nähert er sich 1, war das "
                    "bisher nahe an großen Zyklus-Hochs.",
    "Halving-Phase": "Tage seit dem letzten Bitcoin-Halving. Bisher kamen die Hochs etwa 1–1,5 Jahre und die Tiefs etwa "
                     "2,5 Jahre danach. Nur bereits stattgefundene Halvings zählen. Achtung: nur 3–4 Zyklen als Erfahrung.",
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


def bitstamp_daily(pair="btcusd"):
    rows, start = [], 1313000000          # Bitstamp-Daten beginnen im August 2011
    for _ in range(12):
        d = http_get(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/", {"step": 86400, "limit": 1000, "start": start})
        o = (d.get("data") or {}).get("ohlc") or []
        if not o:
            break
        rows += [(int(x["timestamp"]), x["open"], x["high"], x["low"], x["close"], x["volume"]) for x in o]
        if len(o) < 1000:
            break
        start = int(o[-1]["timestamp"]) + 86400
        time.sleep(0.3)
    return to_df(rows) if rows else None


def blockchain_daily():
    """Durchschnittlicher Bitcoin-Tageskurs (Blockchain.com) – nur Schlusskurse, daher flache Kerzen ohne Volumen."""
    d = http_get("https://api.blockchain.info/charts/market-price",
                 {"timespan": "all", "format": "json", "sampled": "false"}, timeout=40)
    rows = [((int(v["x"]) // 86400) * 86400, float(v["y"]), float(v["y"]), float(v["y"]), float(v["y"]), np.nan)
            for v in d.get("values", []) if float(v["y"]) > 0]
    return to_df(rows) if rows else None


def bitfinex_daily(sym):
    for s in (f"t{sym}USD", f"t{sym}:USD"):
        try:
            d = http_get(f"https://api-pub.bitfinex.com/v2/candles/trade:1D:{s}/hist",
                         {"limit": 10000, "sort": 1}, tries=2)
        except Exception:
            continue
        finally:
            time.sleep(2.2)              # Bitfinex: höchstens 30 Abfragen pro Minute
        if isinstance(d, list) and d and isinstance(d[0], list):
            return to_df([(int(k[0]) // 1000, k[1], k[3], k[4], k[2], k[5]) for k in d])   # MTS, O, C, H, L, V
    return None


def splice(base, older, need_overlap=True):
    """Hängt ältere Kerzen vor die vorhandenen – nur, wenn die Kurse im Überlappungszeitraum zusammenpassen."""
    if older is None or older.empty or base is None or base.empty:
        return base, 0
    first = int(base["t"].iloc[0])
    ov = older[(older["t"] >= first) & (older["t"] < first + 90 * 86400)].merge(base[["t", "c"]], on="t", suffixes=("", "_b"))
    if len(ov) >= 10:
        ratio = float(np.median(ov["c"] / ov["c_b"]))
        if not 0.9 < ratio < 1.1:
            return base, 0
    elif need_overlap:
        return base, 0
    old = older[older["t"] < first]
    if old.empty:
        return base, 0
    return pd.concat([old, base], ignore_index=True), len(old)


def regroup(daily, how):
    """Tageskerzen zu Wochen (Montag) oder Monaten zusammenfassen, mit sauberem Startzeitpunkt."""
    dt = pd.to_datetime(daily["t"], unit="s", utc=True)
    if how == "w":
        key = (daily["t"] - 4 * 86400) // (7 * 86400)
        start = key * 7 * 86400 + 4 * 86400
    else:
        key = dt.dt.year * 12 + dt.dt.month - 1
        start = pd.to_datetime(pd.DataFrame({"year": key // 12, "month": key % 12 + 1, "day": 1})).astype("int64") // 10**9
    g = daily.assign(k=np.asarray(key), s=np.asarray(start)).groupby("k", sort=True)
    return pd.DataFrame({"t": g["s"].first().astype("int64"), "o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(),
                         "c": g["c"].last(), "v": g["v"].sum(min_count=1)}).reset_index(drop=True)


def extend_history(sym, rank, frames):
    """Verlängert Tages-, Wochen- und Monatskerzen um ältere Daten anderer Quellen."""
    d = frames.get("1d")
    if d is None or d.empty:
        return []
    used = []
    try:
        if sym == "BTC":
            d, n = splice(d, bitstamp_daily(), need_overlap=False)
            if n:
                used.append(f"Bitstamp ({n} Tage)")
            d, n = splice(d, blockchain_daily(), need_overlap=True)
            if n:
                used.append(f"Blockchain.com ({n} Tage)")
        elif rank <= EXTEND_RANK and int(d["t"].iloc[0]) < 1577836800:    # nur ältere Coins (auf Binance vor 2020)
            d, n = splice(d, bitfinex_daily(sym), need_overlap=True)
            if n:
                used.append(f"Bitfinex ({n} Tage)")
    except Exception as e:
        log(f"  {sym}: ältere Historie nicht verfügbar ({e})")
    if not used:
        return []
    frames["1d"] = d
    for tf, how in (("1w", "w"), ("1M", "m")):
        cur = frames.get(tf)
        agg = regroup(d, how)
        if cur is None or cur.empty:
            frames[tf] = agg
        else:
            frames[tf] = pd.concat([agg[agg["t"] < int(cur["t"].iloc[0])], cur], ignore_index=True)
    return used


def load_all(sym, rank=9999):
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
        extra = extend_history(sym, rank, frames)
        if extra:
            name = name + " + " + " + ".join(extra)
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
    for key, sources in (("eurusd", [lambda: fred_daily("DEXUSEU"), lambda: yahoo_daily("EURUSD=X")]),
                         ("dxy", [lambda: yahoo_daily("DX-Y.NYB"), lambda: fred_daily("DTWEXBGS")]),
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


HALVING_TS = [int(datetime.strptime(x, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for x in HALVINGS]


def btc_long_scores(btc_daily):
    """Langfristige Bitcoin-Bewertungen (Gruppe „Zyklus“) – je Tag nur mit Daten bis zu diesem Tag."""
    out = {}
    if btc_daily is None or len(btc_daily) < 400:
        return out
    t = btc_daily["t"].values
    c = pd.Series(btc_daily["c"].values, dtype=float)
    avail = t + 86400
    s1400 = c.rolling(1400, min_periods=1400).mean()
    r200w = c / s1400
    out["BTC 200-Wochen-Ø"] = (avail, lin(np.log2(r200w), 0.0, 2.0).values, r200w.values,
                              lambda v: "nahe/unter 200W-Ø – historisch günstig" if v < 1.3 else "weit darüber – historisch teuer"
                              if v > 3 else "dazwischen", lambda v: f"{v:.2f}×", r200w.values)
    mm = c / c.rolling(200, min_periods=200).mean()
    out["BTC Mayer-Multiple"] = (avail, lin(mm, 0.8, 2.4).values, mm.values,
                                 lambda v: "günstig" if v < 0.9 else "überhitzt" if v > 2.2 else "normal",
                                 lambda v: f"{v:.2f}", mm.values)
    pi = c.rolling(111, min_periods=111).mean() / (2 * c.rolling(350, min_periods=350).mean())
    out["BTC Pi-Cycle"] = (avail, lin(pi, 0.45, 1.0).values, pi.values,
                           lambda v: "nahe Pi-Cycle-Top" if v > 0.9 else "weit weg vom Top" if v < 0.6 else "neutral",
                           lambda v: f"{v:.2f}", pi.values)
    days = np.full(len(t), np.nan)
    for h in HALVING_TS:
        days = np.where(t >= h, (t - h) / 86400, days)
    ph = days / 1460
    f = -np.exp(-((ph - 0.33) / 0.09) ** 2) + np.exp(-((ph - 0.62) / 0.10) ** 2)
    out["Halving-Phase"] = (avail, 50 + 50 * f, days,
                            lambda v: f"{v / 365:.1f} J nach Halving – " + ("typische Hoch-Phase" if 350 < v < 620 else
                                                                           "typische Tief-Phase" if 760 < v < 1030 else "Übergang"),
                            lambda v: f"{v:.0f} T", days)
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
    for grp in ("Lage", "Wende", "Markt", "Zyklus"):
        cols = [x for x in subs.columns if GROUP_OF.get(x) == grp]
        if not cols:
            g[grp] = np.full(len(subs), np.nan)
            continue
        w = np.array([weights.get(x, 1.0) for x in cols])
        part = subs[cols].values
        mask = ~np.isnan(part)
        with np.errstate(invalid="ignore", divide="ignore"):
            g[grp] = np.where(mask.any(axis=1), np.nansum(part * w, axis=1) / (mask * w).sum(axis=1), np.nan)
    return dict(C, lage=g["Lage"], wende=g["Wende"], markt=g["Markt"], zyklus=g["Zyklus"])


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


def to_score(raw, S, zy=None):
    """Gestaffelter Score: normale Wendepunkte erreichen etwa 70–80 (bzw. 20–30). Nur wenn zusätzlich die
    langfristigen Zyklus-Indikatoren in dieselbe Richtung zeigen, geht er bis über 90 (bzw. unter 10)."""
    d = raw * S
    n = 35 * np.tanh(d / 0.6 * 0.9)
    with np.errstate(invalid="ignore"):
        L = np.nan_to_num((zy - 50) / 50) if zy is not None else np.zeros(len(d))
        same = np.sign(L) == np.sign(d)
        bonus = np.where(same, 15 * np.abs(L) * np.clip(np.abs(d) / 0.3, 0, 1) * np.sign(d), 5 * L)
    return np.clip(50 + n + bonus, 0, 100)


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
        bn, sn, rets, lows, cbn, csn = [], [], [], 0, [], []
        for r, it in zip(raws, items):
            sc = to_score(r, S, it["C"].get("zyklus"))
            mask = it[part]
            buys, sells = signals(np.where(mask, sc, np.nan))
            n = len(sc)
            bn += nearness(buys, it["lo"][N], n, N)
            sn += nearness(sells[0], it["hi"][N], n, N)
            lo_in = it["lo"][N]
            lows += int(np.sum(mask[lo_in])) if len(lo_in) else 0
            c = it["C"]["c"]
            rets += [c[i + STAT_HORIZON] / c[i] - 1 for i in buys if i + STAT_HORIZON < n]
            if "cyc" in it:
                clo, chi, NC = it["cyc"]
                clo_in, chi_in = clo[mask[clo]] if len(clo) else clo, chi[mask[chi]] if len(chi) else chi
                if len(clo_in):
                    cbn += nearness(buys, clo_in, n + NC, NC) or [0.0]
                if len(chi_in):
                    csn += nearness(sells[0] + sells[1] + sells[2], chi_in, n + NC, NC) or [0.0]
        nb = len(bn)
        # Ziel: Kaufsignale möglichst nah am echten Tief, möglichst wenige; Verkauf halb gewichtet
        obj = (np.mean(bn) if nb else 0) - 0.05 * nb / max(lows, 1) + 0.5 * (np.mean(sn) if sn else 0)
        if cbn or csn:   # Langfristig: zusätzlich Nähe zu den fest hinterlegten Bitcoin-Zyklus-Hochs/-Tiefs
            obj += 0.5 * (np.mean(cbn) if cbn else 0) + 0.25 * (np.mean(csn) if csn else 0)
        res[part] = {"n": nb, "near": float(np.mean(bn) * 100) if nb else 0,
                     "hit": float(np.mean(np.array(rets) > 0) * 100) if rets else 0,
                     "per_low": round(nb / max(lows, 1), 2), "obj": float(obj),
                     "cyc": float(np.mean(cbn) * 100) if cbn else None}
    return S, res


def event_study(items, mode, base_weights):
    """Ereignisstudie: Wie standen die einzelnen Indikatoren direkt VOR großen Anstiegen bzw. Rückgängen?
    Nur Trainingsdaten. Indikatoren, die vorher zuverlässig richtig standen, bekommen mehr Gewicht."""
    H = 20 if mode == "short" else 30
    up, dn = {}, {}
    n_up = n_dn = 0
    for it in items:
        C = it["C"]
        c = C["c"]
        n = len(c)
        if n <= H + 50:
            continue
        fwd = np.full(n, np.nan)
        fwd[:-H] = c[H:] / c[:-H] - 1
        fwd[~it["train"]] = np.nan
        ok = ~np.isnan(fwd)
        if ok.sum() < 50:
            continue
        q_up, q_dn = np.nanquantile(fwd[ok], 0.9), np.nanquantile(fwd[ok], 0.1)

        def starts(cond):
            idx, last = [], -10 ** 9
            for i in np.where(cond)[0]:
                if i - last >= H:
                    idx.append(i)
                    last = i
            return np.array(idx, dtype=int)
        # Für die Messung zählen ALLE Kerzen vor einem großen Anstieg/Rückgang (oberstes/unterstes Zehntel). Würde man
        # nur den „Beginn“ jedes Anstiegs nehmen, wäre dieser Punkt mit Blick in die Zukunft ausgewählt und läge
        # automatisch an einem kleinen Tief – das würde Vorlauf vortäuschen.
        iu, idn = np.where(ok & (fwd >= q_up))[0], np.where(ok & (fwd <= q_dn))[0]
        n_up += len(starts(ok & (fwd >= q_up)))
        n_dn += len(starts(ok & (fwd <= q_dn)))
        for col in C["subs"].columns:
            if GROUP_OF.get(col) == "Zyklus":
                continue
            v = C["subs"][col].values
            up.setdefault(col, []).extend(v[iu].tolist())
            dn.setdefault(col, []).extend(v[idn].tolist())
    seps, weights = {}, dict(base_weights)
    for col in up:
        a, b = np.array(up[col], float), np.array(dn[col], float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        sd = np.sqrt((a.var() + b.var()) / 2) + 1e-9
        sep = float((a.mean() - b.mean()) / sd)
        seps[col] = round(sep, 3)
        weights[col] = round(base_weights.get(col, 1.0) * float(np.clip(1 + 2 * sep, 0.25, 2.5)), 3)
    log(f"Ereignisstudie {mode}: {n_up} Anstiege, {n_dn} Rückgänge; beste Vorläufer: "
        + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(seps.items(), key=lambda x: -x[1])[:4]))
    return weights, seps, {"up": n_up, "dn": n_dn, "H": H}


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

    def approach(fast, slow):
        """+1 = schneller Durchschnitt nähert sich von unten (bullisches Kreuz naht), -1 = von oben (bärisch)."""
        g = (fast - slow) / slow
        dg = (g - g.shift(5)) / 5
        with np.errstate(invalid="ignore", divide="ignore"):
            eta = -g / dg.replace(0, np.nan)
        conv = (np.sign(g) != np.sign(dg)) & eta.notna() & (eta >= 0)
        s = np.where(conv, -np.sign(g) * np.exp(-eta.clip(lower=0) / 20), 0.0)
        crossed = np.sign(g) != np.sign(g.shift(5))           # gerade gekreuzt
        s = np.where(crossed & g.notna() & g.shift(5).notna(), np.sign(g) * 0.8, s)
        return pd.Series(s).where(g.notna() & dg.notna())
    cross = pd.concat([approach(ema(c, 20), sma(c, 50)), approach(sma(c, 50), sma(c, 200))], axis=1).mean(axis=1)

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
        "Kreuz naht": (50 + 50 * cross, cross, lambda x: f"{x:+.2f}",
                       lambda x: "bullisches Kreuz naht" if x > 0.3 else "bärisches Kreuz naht" if x < -0.3 else "kein Kreuz in Sicht"),
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


def rnd(arr):
    return [None if np.isnan(x) else round(float(x), 1) for x in arr]


def tf_output(df, per_mode, details, extra, overall):
    out = {"c": [[int(r.t), sig(r.o), sig(r.h), sig(r.l), sig(r.c)] for r in df.itertuples()]}
    n = len(df)
    for mode, M in MODES.items():
        sfx = "" if mode == "short" else "l"
        N = M["pivN"] if n >= 3 * M["pivN"] else max(3, n // 8)
        hi, lo = pivot_idx(df["h"].values, df["l"].values, N)
        if overall.get(mode) is not None and np.isfinite(overall[mode]).any():
            out["zg" + sfx] = rnd(overall[mode])
        pm = per_mode.get(mode)
        if pm is None:
            continue
        score, groups = pm
        out["z" + sfx] = rnd(score)
        buys, sells = signals(score)
        out["sg" + sfx] = [[i, "B"] for i in buys] + [[i, f"S{k}"] for k, s in enumerate(sells, 1) for i in s]
        out["st" + sfx] = mode_stats(df["c"].values, buys, sells, lo, N)
        out["k" + sfx] = groups
    if details:
        out["b"] = details
    out.update(extra)
    return out


OVERALL_K = {"short": 1.0, "long": 1.0}


def overall_at(ct, mode, scores):
    """Gesamt-Score zu den Zeitpunkten ct aus den zuletzt abgeschlossenen Kerzen mehrerer Zeitrahmen."""
    num = np.zeros(len(ct))
    den = np.zeros(len(ct))
    for tf, w in OVERALL[mode].items():
        if tf not in scores:
            continue
        sct, sv = scores[tf]
        v = take(sv, np.searchsorted(sct, ct, side="right") - 1)
        ok = ~np.isnan(v)
        num[ok] += w * v[ok]
        den[ok] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    # Mitteln mehrerer Zeitrahmen drückt den Wert Richtung 50 – Spreizung wieder angleichen
    return np.clip(50 + (avg - 50) * OVERALL_K[mode], 0, 100)


# Fest hinterlegte Hochs und Tiefs der Bitcoin-Vier-Jahres-Zyklen (Tageshoch/-tief, Binance bzw. Bitstamp vor 2017)
BTC_CYCLE = [("2013-12-04", "H", 1163), ("2015-01-14", "T", 152),
             ("2017-12-17", "H", 19798), ("2018-12-15", "T", 3156),
             ("2021-11-10", "H", 69000), ("2022-11-21", "T", 15476),
             ("2025-10-06", "H", 126200)]


def btc_cycle_points(btc_daily):
    """Feste Zyklus-Punkte. Das Tief nach dem letzten Hoch kommt erst dazu, wenn es sich in den Daten bestätigt
    (Kurs seitdem mindestens +50 % und das Tief mindestens 180 Tage alt)."""
    ts = lambda d: int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    pts = [[ts(d), k, p, True] for d, k, p in BTC_CYCLE]
    if btc_daily is not None and len(btc_daily) > 30 and BTC_CYCLE[-1][1] == "H":
        t, l, c = btc_daily["t"].values, btc_daily["l"].values, btc_daily["c"].values
        after = np.where(t > ts(BTC_CYCLE[-1][0]))[0]
        if len(after):
            il = after[np.argmin(l[after])]
            if c[-1] >= 1.5 * l[il] and time.time() - t[il] > 180 * 86400:
                pts.append([int(t[il]), "T", sig(l[il]), True])
    return pts


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
        src, frames = load_all(sym, meta.get("rank") or 9999)
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
    mscores.update(btc_long_scores(btc.get("1d") if btc else None))
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
            cal_tfs = ("1d", "1w") if mode == "long" else ("4h", "1d")
            cyc_ts = np.array([int(datetime.strptime(d_, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
                               for d_, k_, p_ in BTC_CYCLE])
            cyc_k = np.array([k_ for d_, k_, p_ in BTC_CYCLE])

            def build_items(W):
                items = []
                gi = {tf: with_groups(C, W) for tf, C in idx_C.items()}
                for cc in comps[:CALIB_COINS]:
                    g = {tf: with_groups(C, W) for tf, C in cc.items() if tf in ("4h", "1d", "1w", "1M")}
                    for tf in cal_tfs:
                        if tf not in g:
                            continue
                        C = g[tf]
                        H = g.get(HTF[tf])
                        Mi = gi.get(tf)
                        n = len(C["c"])
                        train = np.zeros(n, bool)
                        train[:int(n * 0.7)] = True
                        hi, lo = pivot_idx(C["h"], C["l"], M["pivN"])
                        it = {"C": C, "H": H, "hidx": htf_map(C, H) if H else None,
                              "M": Mi, "midx": exact_index(Mi["t"], C["t"]) if Mi else None,
                              "train": train, "test": ~train, "lo": {M["pivN"]: lo}, "hi": {M["pivN"]: hi}}
                        if mode == "long":
                            tt = C["t"]
                            inside = (cyc_ts >= tt[0]) & (cyc_ts <= tt[-1])
                            pos = np.searchsorted(tt, cyc_ts, side="right") - 1
                            NC = max(4, int(round(30 * 86400 / TF_SEC[tf])))
                            it["cyc"] = (pos[inside & (cyc_k == "T")], pos[inside & (cyc_k == "H")], NC)
                        items.append(it)
                return items

            items_base = build_items(M["weights"])
            if not items_base:
                calib["modes"][mode] = {"cfg": M["default"], "S": 2.0, "res": None, "default": None, "adopted": False,
                                        "weights": M["weights"], "wset": "Standard"}
                continue
            ew, seps, ev = event_study(items_base, mode, M["weights"])
            cb = calibrate(items_base, mode)
            ce = calibrate(build_items(ew), mode)
            # Ereignis-Gewichte nur übernehmen, wenn sie auch im unbekannten Testzeitraum mindestens gleich gut sind
            ok_e = ce["res"] and cb["res"] and ce["res"]["test"]["obj"] >= cb["res"]["test"]["obj"] and ce["res"]["test"]["n"] >= 8
            best = ce if ok_e else cb
            best["weights"] = ew if ok_e else M["weights"]
            best["wset"] = "Ereignisstudie" if ok_e else "Standard"
            best["events"] = ev
            best["seps"] = seps
            best["ew"] = ew
            best["base_test"] = cb["res"]["test"] if cb["res"] else None
            log(f"Gewichte {mode}: {best['wset']}")
            calib["modes"][mode] = best
    log(f"Kalibrierung fertig nach {time.time() - started:.0f}s")

    # 5) Endgültige Scores + Ausgabe
    mode_idx = {}
    WM = {mode: calib["modes"][mode].get("weights") or M["weights"] for mode, M in MODES.items()}
    for mode, M in MODES.items():
        cfg = calib["modes"][mode]["cfg"]
        mode_idx[mode] = {tf: combine(with_groups(C, WM[mode]), cfg)[0] for tf, C in idx_C.items()}
    listing = []
    prepared = []
    for (meta, src, frames), cc in zip(loaded, comps):
        gm = {mode: {tf: with_groups(C, WM[mode]) for tf, C in cc.items()} for mode in MODES}
        bases = {mode: {tf: combine(C, calib["modes"][mode]["cfg"])[0] for tf, C in gm[mode].items()} for mode in MODES}
        per_tf = {tf: {} for tf in frames}
        extras = {tf: {} for tf in frames}
        scores = {mode: {} for mode in MODES}
        for tf, C in cc.items():
            htf = HTF.get(tf)
            for mode in MODES:
                cal = calib["modes"][mode]
                cfg, S = cal["cfg"], cal["S"]
                Cg = gm[mode][tf]
                hb = take(bases[mode][htf], htf_map(C, cc[htf])) if htf in cc else None
                mb = take(mode_idx[mode][tf], exact_index(idx_C[tf]["t"], C["t"])) if tf in idx_C else None
                score = to_score(combine(Cg, cfg, hb, mb)[1], S, Cg["zyklus"])
                last = len(score) - 1
                groups = {g: (None if np.isnan(Cg[k][last]) else round(float(Cg[k][last])))
                          for g, k in (("Lage", "lage"), ("Wende", "wende"), ("Markt", "markt"), ("Zyklus", "zyklus"))}
                per_tf[tf][mode] = (score, groups)
                scores[mode][tf] = (C["ct"], score)
                if mode == "short":
                    hv = to_score(hb, S)[-1] if hb is not None else np.nan
                    mv = to_score(mb, S)[-1] if mb is not None else np.nan
                    extras[tf]["f"] = {"htf": [htf, None if np.isnan(hv) else round(float(hv))] if htf in cc else None,
                                       "mkt": None if np.isnan(mv) else round(float(mv))}
        prepared.append((meta, src, frames, cc, gm, per_tf, extras, scores))

    # Spreizung des Gesamt-Scores so wählen, dass er so stark schwankt wie ein einzelner Zeitrahmen
    for mode in MODES:
        OVERALL_K[mode] = 1.0
        base_tf = max(OVERALL[mode], key=OVERALL[mode].get)
        ratios = []
        for p in prepared[:40]:
            sc = p[7][mode]
            if base_tf not in sc:
                continue
            ct, comp = sc[base_tf]
            ov = overall_at(ct, mode, sc)
            ok = ~np.isnan(comp) & ~np.isnan(ov)
            if ok.sum() > 100 and np.std(ov[ok]) > 0:
                ratios.append(np.std(comp[ok]) / np.std(ov[ok]))
        OVERALL_K[mode] = float(np.clip(np.median(ratios), 1.0, 2.5)) if ratios else 1.0
        log(f"Gesamt-Score {mode}: Spreizung {OVERALL_K[mode]:.2f}")

    for meta, src, frames, cc, gm, per_tf, extras, scores in prepared:
        tfs = {}
        for tf, _ in TIMEFRAMES:
            df = frames.get(tf)
            if df is None:
                continue
            C = cc.get(tf)
            extra, details = extras.get(tf, {}), None
            if C is not None:
                last = len(df) - 1
                details = []
                for name, v in C["ind"].items():
                    sv = float(np.asarray(v[0], dtype=float)[last])
                    rv = float(np.asarray(v[1], dtype=float)[last])
                    if np.isnan(sv):
                        continue
                    details.append([name, GROUP_OF.get(name, "Markt"), round(sv),
                                    "–" if np.isnan(rv) else v[2](rv), "–" if np.isnan(rv) else v[3](rv)])
                extra["cyp"] = C["cyc_per"]
                pane = {}
                for name, v in C["ind"].items():
                    series = v[4] if len(v) > 4 else v[1]
                    pane[name] = [r3(x) for x in np.asarray(series, dtype=float)]
                for g, k in (("Score-Teil Lage", "lage"), ("Score-Teil Wende", "wende"), ("Score-Teil Markt", "markt"),
                             ("Score-Teil Zyklus", "zyklus")):
                    pane[g] = [r3(x) for x in gm["short"][tf][k]]
                with open(os.path.join(OUT_DIR, "data", "ind", f"{safe_name(meta['sym'])}_{tf}.json"), "w") as f:
                    json.dump({"t0": int(df["t"].iloc[0]), "v": pane}, f, separators=(",", ":"))
                extra["ind"] = 1
            ct = close_times(df["t"].values)
            overall = {mode: overall_at(ct, mode, scores[mode]) for mode in MODES}
            tfs[tf] = tf_output(df, per_tf.get(tf, {}), details, extra, overall)
        fname = safe_name(meta["sym"])
        with open(os.path.join(OUT_DIR, "data", fname + ".json"), "w", encoding="utf-8") as f:
            json.dump({"s": meta["sym"], "src": src, "tf": tfs}, f, separators=(",", ":"))
        far = np.array([2 ** 40])
        g_now = {mode: overall_at(far, mode, scores[mode])[0] for mode in MODES}
        listing.append({"s": meta["sym"], "n": meta["name"], "r": meta["rank"], "f": fname, "src": src,
                        "p": sig(meta["price"] or frames["4h"]["c"].iloc[-1]),
                        "ch": None if meta["chg"] is None else round(float(meta["chg"]), 2),
                        "g": None if np.isnan(g_now["short"]) else round(float(g_now["short"]), 1),
                        "gl": None if np.isnan(g_now["long"]) else round(float(g_now["long"]), 1),
                        "sc": {tf: last_valid(tfs.get(tf, {}).get("z")) for tf in LIST_TFS},
                        "scl": {tf: last_valid(tfs.get(tf, {}).get("zl")) for tf in LIST_TFS}})
    listing.sort(key=lambda c: c["r"])

    mk = {}
    if "fng" in macro:
        mk["fng"] = [float(macro["fng"][0].iloc[-1]), macro.get("fng_label", "")]
    if "dxy" in macro:
        mk["dxy"] = round(float(macro["dxy"][0].iloc[-1]), 2)
    fx = []
    if "eurusd" in macro:
        s = macro["eurusd"][0]
        s = s[s.index >= 1262304000]            # ab 2010
        fx = [[int(t) // 86400, round(float(v), 4)] for t, v in s.items()]
    if "global" in macro:
        mk.update(macro["global"])
    payload = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "buy": BUY_LEVEL, "sells": SELL_LEVELS, "reset": RESET_LEVEL, "h": STAT_HORIZON, "tfs": TIMEFRAMES,
               "ltfs": LIST_TFS, "info": IND_INFO, "go": GROUP_OF,
               "modes": {m: {"label": M["label"], "pivN": M["pivN"], "w": WM[m]} for m, M in MODES.items()},
               "basew": {m: M["weights"] for m, M in MODES.items()},
               "cal": calib, "mkt": mk, "overall": OVERALL, "overall_k": OVERALL_K,
               "btcc": btc_cycle_points(btc.get("1d") if btc else None), "fx": fx, "coins": listing}
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
.cw{position:relative;user-select:none;-webkit-user-select:none}
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
@media (orientation:landscape) and (max-height:560px) and (max-width:999px){
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
.touch.pany{touch-action:pan-y}
.pscroll{position:absolute;top:0;right:0;width:78px;z-index:7;touch-action:pan-y}
.kl{display:flex;gap:5px}
.sc2{min-width:42px;text-align:center;font-weight:800;font-size:16px;padding:3px 0 4px;border-radius:9px;opacity:.55;line-height:1.1}
.sc2.act{opacity:1}
.sc2 small{display:block;font-size:9px;font-weight:700;opacity:.8}
.opts{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 8px 4px}
.chip.on[style*="--c"]{background:var(--c);border-color:var(--c);color:#1a1405}
.thr{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--mut)}
.thr input{width:92px;accent-color:var(--acc)}
.thr b{font-weight:600;min-width:64px;font-variant-numeric:tabular-nums}
.ptag{position:absolute;transform:translateX(-50%);font-size:10.5px;font-weight:700;padding:1px 5px;border-radius:5px;
white-space:nowrap;font-variant-numeric:tabular-nums}
.ptag.H{color:#fecaca;background:#7f1d1dcc}.ptag.T{color:#bbf7d0;background:#14532dcc}
.vl.cyc{border-left:2px solid #eab308;opacity:.85}
.ctag{position:absolute;top:2px;transform:translateX(4px);font-size:10px;line-height:1.25;color:#1a1405;background:#eab308e6;
padding:1px 5px;border-radius:5px;white-space:nowrap;font-weight:700}
.bub.sm{padding:3px 7px;font-size:11px}.bub.sm b{font-size:12px}
.bub.mid{transform:translate(-50%,-50%);padding:3px 8px}
@media (orientation:landscape) and (max-height:560px) and (max-width:999px){ .opts{margin:0 4px 2px} }
.bub i{position:absolute;right:3px;top:1px;font-style:normal;font-size:9px;opacity:.55}
.bub.sm{padding-right:14px}
.bub.grab{box-shadow:0 0 0 2px var(--acc),0 4px 14px #0008}
.ctag{overflow:hidden;text-overflow:ellipsis;transform:none}
.vl.cyc.T{border-left-color:#22d3ee}
.ctag.T{background:#22d3eee6;color:#04212a}
.chip.on[style*="--c"] span{color:#1a1405!important}
.chip.ma i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:1px}
.chip.ma.on{background:var(--card2);color:var(--txt);border-color:var(--acc)}
li.row.sel{outline:2px solid var(--acc)}
@media (min-width:1000px){
  header,.stats,#list,body>.sec,footer{width:400px}
  body>.sec{margin-left:10px!important;width:380px}
  #detail{display:block;left:400px;border-left:1px solid var(--line)}
  #detail .back{display:none}
  #tfc{flex-wrap:wrap}
  #dmain{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));align-items:start}
  #dmain .box,#dmain .note,#dmain .sec.exp,#dmain #allbox{grid-column:1/-1}
  .opts{gap:8px}
}
.simf{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.simf label{display:flex;flex-direction:column;font-size:11.5px;color:var(--mut);gap:3px}
.simf input,.simf select,.trs select{background:var(--card2);color:var(--txt);border:1px solid var(--line);border-radius:8px;padding:7px 8px;font-size:14px;font-family:inherit}
.trs{display:inline-flex;align-items:center;gap:4px;font-size:12px}
.trs select{padding:3px 4px;font-size:12px}
.tradew{max-height:420px;overflow:auto;margin-top:6px}
#trades th{font-size:11px;color:var(--mut);text-align:left;padding:4px;position:sticky;top:0;background:var(--card)}
#trades td{font-size:12px;padding:6px 4px;vertical-align:top}
#trades small{color:var(--mut);font-size:10.5px}
.evt td{font-size:11.5px;padding:3px 4px}
</style>
</head>
<body>
<header>
  <div class="hrow"><h1>Krypto Score</h1><button class="fresh" id="ref"></button></div>
  <div class="mbar" id="mbar"></div>
  <input class="search" id="q" type="search" placeholder="Coin suchen (z. B. BTC, Solana)…" autocomplete="off">
  <div class="chips"><span class="lbl">Modus:</span><div class="seg mseg"><button data-m="short">Kurzfristig</button><button data-m="long">Langfristig</button></div>
    <span class="lbl" style="margin-left:6px">Währung:</span><div class="seg cseg"><button data-c="USD">$</button><button data-c="EUR">€</button></div></div>
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
    <div class="opts" id="opts">
      <button class="chip sm" id="tgPiv">Hochs/Tiefs</button>
      <span class="thr" id="thrw"><input type="range" id="thr" min="3" max="80" step="1"><b id="thrv"></b></span>
      <button class="chip sm" id="tgCyc" style="--c:#eab308">₿-Zyklus <span style="color:#eab308">▲</span><span style="color:#22d3ee">▼</span></button>
      <span class="lbl">Tranchen:</span><span class="trs"><b class="up">Kauf</b><select id="nB"></select><b class="dn">Verk.</b><select id="nS"></select></span>
      <div class="seg cseg"><button data-c="USD">$</button><button data-c="EUR">€</button></div>
      <span id="mas"></span>
      <span class="lbl" style="margin-left:4px">Score:</span><div class="seg" id="zseg"><button data-z="tf">Zeitrahmen</button><button data-z="g">Gesamt</button></div>
    </div>
    <div class="mres" id="mres"></div>
    <div class="cwrap" id="cwrap"></div>
    <div class="msg" id="noz" style="display:none">Für diesen Zeitrahmen gibt es zu wenige Kerzen für einen eigenen Score – schalte auf „Gesamt“.</div>
  </div>
  <div class="sec" id="simbox"><h4>Simulation (Backtest) mit diesen Signalen</h4>
    <div class="simf">
      <label>Start<input type="date" id="simStart"></label>
      <label>Budget<input type="number" id="simBudget" min="1" step="100"></label>
      <label>Produkt<select id="simProd"><option value="spot">Spot (Krypto direkt)</option><option value="etp">ETP (z. B. Scalable)</option><option value="fut">Futures (z. B. Bitget)</option></select></label>
      <label>Gebühr % pro Trade<input type="number" id="simFee" min="0" step="0.01"></label>
      <label class="futonly">Hebel<input type="number" id="simLev" min="1" max="20" step="1"></label>
      <label class="futonly">Finanzierung % p. a.<input type="number" id="simFund" min="0" step="0.5"></label>
      <label class="spotonly">Pers. Steuersatz %<input type="number" id="simTaxP" min="0" max="60" step="0.5"></label>
      <label class="spotonly">Freigrenze € / Jahr<input type="number" id="simFreeP" min="0" step="100"></label>
      <label class="capOnly">Abgeltungsteuer %<input type="number" id="simTaxC" min="0" max="60" step="0.005"></label>
      <label class="capOnly">Sparerpauschbetrag €<input type="number" id="simFreeC" min="0" step="100"></label>
    </div>
    <div class="grid3" id="simres" style="margin-top:10px"></div>
    <p class="note" id="simnote" style="margin:8px 0 0"></p>
    <details style="margin-top:8px"><summary>Trade-Übersicht</summary><div class="tradew"><table id="trades"></table></div></details>
  </div>
  <div class="sec" id="statbox"><h4>Bilanz der Signale (gesamte Historie, Standard-Tranchen)</h4><div class="grid2" id="stat"></div><div class="grid4" id="stat2"></div></div>
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
let filter = 'all', query = '', ltf = store('ltf2') || 'g', ctf = store('ctf') || '4h', sortBy = store('sort') || 'mc';
let mode = store('mode') === 'long' ? 'long' : 'short';
let ccy = store('ccy') === 'EUR' ? 'EUR' : 'USD';
let nB = Math.min(5, Math.max(1, +(store('nB') || 1))), nS = Math.min(5, Math.max(1, +(store('nS') || 3)));
const buyLv = () => Array.from({length: nB}, (_, k) => 80 + k * (nB > 1 ? Math.min(5, 15 / (nB - 1)) : 0));
const sellLv = () => Array.from({length: nS}, (_, k) => 30 - k * (nS > 1 ? Math.min(10, 25 / (nS - 1)) : 0));
const FX = D.fx || [];
function fxAt(t){ if (!FX.length) return null; const d = Math.floor(t / 86400); let lo = 0, hi = FX.length - 1;
  if (d <= FX[0][0]) return FX[0][1]; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (FX[m][0] <= d) lo = m; else hi = m - 1; } return FX[lo][1]; }
const fxNow = () => FX.length ? FX[FX.length - 1][1] : null;
const toC = (usd, t) => { if (usd == null || ccy !== 'EUR') return usd; const f = t == null ? fxNow() : fxAt(t); return f ? usd / f : usd; };
let mas = []; try { mas = JSON.parse(store('mas') || '[]') || []; } catch (e) { mas = []; }
const MA = [['e20', 'EMA 20', '#60a5fa'], ['s50', 'SMA 50', '#f59e0b'], ['s200', 'SMA 200', '#a78bfa'], ['w200', '200W-Ø', '#f472b6']];
let showPiv = store('piv') !== '0', showCyc = store('cyc') !== '0', zsel = store('zsel') === 'tf' ? 'tf' : 'g';
const thrGet = () => +(store('thr_' + mode) || (mode === 'long' ? 40 : 20));
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
  const s = p.toLocaleString('de-DE', {minimumFractionDigits: d, maximumFractionDigits: d});
  return ccy === 'EUR' ? s + ' €' : '$' + s;
}
const money = p => p == null || !isFinite(p) ? '–' : (ccy === 'EUR' ? Math.round(p).toLocaleString('de-DE') + ' €' : '$' + Math.round(p).toLocaleString('de-DE'));
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
const landscape = () => window.matchMedia('(orientation: landscape) and (max-height: 560px) and (max-width: 999px)').matches;
const desktop = () => window.matchMedia('(min-width: 1000px)').matches;

// ---------------- Kopfzeile: Markt + Frische
function mbar(){
  const m = D.mkt || {}, out = [];
  const fgc = v => v < 25 ? '#ef4444' : v < 45 ? '#f97316' : v <= 55 ? '#eab308' : v <= 75 ? '#84cc16' : '#22c55e';
  if (m.fng) out.push(`<div class="mb">Fear&Greed <b style="color:${fgc(m.fng[0])}">${Math.round(m.fng[0])}</b> ${esc(m.fng[1])}</div>`);
  if (m.dom != null) out.push(`<div class="mb">BTC-Dominanz <b>${m.dom.toFixed(1).replace('.', ',')} %</b></div>`);
  if (m.mcap) out.push(`<div class="mb">Marktkap. <b>${(toC(m.mcap) / 1e12).toFixed(2).replace('.', ',')} Bio. ${ccy === 'EUR' ? '€' : '$'}</b> <span class="${m.mcap_ch >= 0 ? 'up' : 'dn'}">${pct(m.mcap_ch)}</span></div>`);
  if (m.dxy) out.push(`<div class="mb">DXY <b>${m.dxy.toFixed(1).replace('.', ',')}</b></div>`);
  $('mbar').innerHTML = out.join('');
  if (!out.length) $('mbar').style.display = 'none';
}
mbar();
function freshness(){
  const t = Math.max(Date.parse(D.updated), liveTs), min = (Date.now() - t) / 60000;
  const txt = min < 1 ? 'gerade eben' : min < 60 ? `vor ${Math.round(min)} Min` : `vor ${Math.floor(min / 60)} Std ${Math.round(min % 60)} Min`;
  const col = min < 10 ? 'var(--buy)' : min < 70 ? '#eab308' : 'var(--sell)';
  $('ref').innerHTML = `<i style="background:${col}"></i>Preise ${txt} <span>⟳</span>`;
  $('ref2').innerHTML = `<i style="background:${col}"></i>${min < 1 ? 'jetzt' : min < 60 ? Math.round(min) + ' Min' : Math.floor(min / 60) + ' Std'} <span>⟳</span>`;
}
freshness(); setInterval(freshness, 15000);

// ---------------- Währung und Tranchen
function setCcy(c){
  if (c === 'EUR' && !FX.length) { alert('Wechselkurse gerade nicht verfügbar.'); return; }
  ccy = c; store('ccy', c);
  document.querySelectorAll('.cseg button').forEach(b => b.classList.toggle('on', b.dataset.c === c));
  mbar(); render(); if (cur && cur.data) show(true);
}
document.querySelectorAll('.cseg').forEach(s => s.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setCcy(b.dataset.c); }));
document.querySelectorAll('.cseg button').forEach(b => b.classList.toggle('on', b.dataset.c === ccy));
['nB', 'nS'].forEach(id => { $(id).innerHTML = [1, 2, 3, 4, 5].map(k => `<option value="${k}">${k}</option>`).join('');
  $(id).value = id === 'nB' ? nB : nS;
  $(id).addEventListener('change', () => { if (id === 'nB') nB = +$(id).value; else nS = +$(id).value; store(id, $(id).value); explain(); if (cur && cur.data) show(true); }); });

// ---------------- Modus
function setMode(m){
  mode = m; store('mode', m);
  document.querySelectorAll('.mseg button').forEach(b => b.classList.toggle('on', b.dataset.m === m));
  render(); explain();
  if (cur && cur.data) show(true);
}
document.querySelectorAll('.mseg').forEach(s => s.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setMode(b.dataset.m); }));

// ---------------- Liste
$('ltf').insertAdjacentHTML('beforeend', `<button class="chip" data-t="g">Gesamt</button>` + D.ltfs.map(t => `<button class="chip" data-t="${t}">${TFL[t]}</button>`).join(''));
const TFN = t => t === 'g' ? 'Gesamt' : TFL[t];
function setLtf(t){ ltf = t; store('ltf2', t); document.querySelectorAll('#ltf .chip').forEach(b => b.classList.toggle('on', b.dataset.t === t)); render(); }
function setSort(o){ sortBy = o; store('sort', o); document.querySelectorAll('#sort .chip').forEach(b => b.classList.toggle('on', b.dataset.o === o)); render(); }
$('ltf').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setLtf(b.dataset.t); });
$('sort').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setSort(b.dataset.o); });
const scoreOf = c => ltf === 'g' ? (mode === 'long' ? c.gl : c.g) : (mode === 'long' ? c.scl : c.sc)[ltf];
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
    `<div class="stat"><b style="color:${color(avg)}">${avg == null ? '–' : avg.toFixed(0)}</b><span>Ø Score (${TFN(ltf)})</span></div>` +
    `<div class="stat"><b class="up">${all.filter(s => s >= BUY).length}</b><span>in Kaufzone</span></div>` +
    `<div class="stat"><b class="dn">${all.filter(s => s <= SELL).length}</b><span>in Verkaufszone</span></div>`;
  const rows = D.coins.filter(match);
  if (sortBy === 'mc') rows.sort((a, b) => a.r - b.r);
  else rows.sort((a, b) => { const x = scoreOf(a), y = scoreOf(b); if (x == null) return 1; if (y == null) return -1; return sortBy === 'hi' ? y - x : x - y; });
  $('list').innerHTML = rows.length ? rows.map(c => {
    const s = scoreOf(c), col = color(s);
    return `<li class="row" data-s="${esc(c.s)}"><div class="rk">#${c.r}</div>
      <div class="nm"><b>${esc(c.s)}</b><small>${esc(c.n)}</small>
        <div class="px">${price(toC(c.p))} <span class="${c.ch >= 0 ? 'up' : 'dn'}">${pct(c.ch)}</span> · ${label(s)}</div>
        <div class="bar"><i style="width:${s || 0}%;background:${col}"></i></div></div>
      <div class="kl"><div class="sc2${mode === 'short' ? ' act' : ''}" style="color:${color(c.g)};background:${color(c.g)}22"><small>K</small>${c.g == null ? '–' : Math.round(c.g)}</div>
        <div class="sc2${mode === 'long' ? ' act' : ''}" style="color:${color(c.gl)};background:${color(c.gl)}22"><small>L</small>${c.gl == null ? '–' : Math.round(c.gl)}</div></div></li>`;
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
    Die Einstellungen wurden an der Historie der Top-40-Coins (${mode === 'long' ? '1 T + 1 W, zusätzlich gemessen an den fest hinterlegten Bitcoin-Zyklus-Hochs und -Tiefs' : '4 Std + 1 T'}) so gewählt, dass Kaufsignale <b>möglichst selten
    und möglichst nah am echten Tief</b> liegen. Eingestellt an den ersten 70 % der Daten, kontrolliert an den letzten 30 %.
    ${cal.adopted ? 'Die neue Einstellung war auch im Kontrollzeitraum besser.' : 'Im Kontrollzeitraum war keine Einstellung besser – es gilt der Standard.'}
    Kontrollzeitraum: <b>${r.test.n} Kaufsignale</b>, Nähe zum Tief <b>${Math.round(r.test.near)} %</b>
    (100 % = genau am Tief, 0 % = ${MD.pivN}+ Kerzen daneben), Gewinn nach ${D.h} Kerzen in <b>${Math.round(r.test.hit)} %</b> der Fälle${d0 ? ` (Standard: ${d0.test.n} Signale, Nähe ${Math.round(d0.test.near)} %)` : ''}.</p>` : '';
  const ev = cal.events, seps = cal.seps || {}, ew = cal.ew || {};
  const top = Object.entries(seps).sort((a, b) => b[1] - a[1]);
  const evTxt = ev ? `<p><b>Ereignisstudie</b> (Vorlauf): In der Historie der Top-Coins wurden ${ev.up} große Anstiege und ${ev.dn} große
    Rückgänge gesucht (oberstes bzw. unterstes Zehntel der Kursänderung über die nächsten ${ev.H} Kerzen, nur Trainingsdaten).
    Für jeden Indikator wurde gemessen, wie deutlich er <i>vorher</i> richtig stand (Trennschärfe: 0 = kein Vorlauf, über 0,5 = deutlich).
    Gute Vorläufer bekamen mehr Gewicht (höchstens das 2,5-Fache), schwache weniger. Verwendet: <b>${esc(cal.wset || 'Standard')}-Gewichte</b>
    ${cal.wset === 'Ereignisstudie' ? '(waren im unbekannten Testzeitraum besser)' : '(die Ereignis-Gewichte waren im Testzeitraum nicht besser)'}.</p>
    <table class="evt">${top.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="v"><span class="mini"><i style="width:${Math.min(100, Math.max(0, v) * 100)}%;background:${v > 0.3 ? 'var(--buy)' : v > 0 ? '#eab308' : 'var(--sell)'}"></i></span> ${v.toFixed(2).replace('.', ',')}</td><td class="v">× ${((ew[k] || 1) / ((D.basew[mode] || {})[k] || 1)).toFixed(2).replace('.', ',')}</td></tr>`).join('')}</table>` : '';
  const html = `
  <p><b>Modus ${esc(MD.label)}</b>: ${mode === 'long' ? 'sucht nur die großen Zyklus-Tiefs und -Hochs (echtes Hoch/Tief = Extrem über ±30 Kerzen), mit mehr Gewicht auf langfristigen Indikatoren und dem übergeordneten Zeitrahmen. Signale sind sehr selten.' : 'sucht die kleineren Tiefs und Hochs (Extrem über ±10 Kerzen) für kürzere Trades.'}</p>
  <p><b>Datenquellen:</b> Kerzen von Binance (Ersatz OKX/KuCoin). Ältere Historie: Bitcoin von Bitstamp (ab 2011) und
  Blockchain.com (Tagesdurchschnittskurse ab 2010, ohne Hoch/Tief/Volumen), ältere Altcoins von Bitfinex – nur übernommen,
  wenn die Kurse im Überlappungszeitraum übereinstimmen.</p>
  <p>Jeder Indikator wird für jede Kerze in einen <b>Teil-Score von 0 bis 100</b> umgerechnet (100 = Kaufen, 0 = Verkaufen):</p>
  <p><b>1. Lage</b> – Wie weit unten oder oben steht der Kurs?</p><ul>${grp('Lage').map(li).join('')}</ul>
  <p><b>2. Markt</b> (${p(cfg.wm / (1 + cfg.wm))} der Richtung) – Stimmung und Umfeld:</p><ul>${grp('Markt').map(li).join('')}</ul>
  <p><b>3. Wende</b> – Gibt es Zeichen, dass der Kurs dreht?</p><ul>${grp('Wende').map(li).join('')}</ul>
  <p><b>4. Bitcoin-Zyklus</b> – langfristige Bewertung aus dem Bitcoin-Kurs (gilt für alle Coins, normales Gewicht):</p><ul>${grp('Zyklus').map(li).join('')}</ul>
  <p><b>Gestaffelter Score:</b> Lage, Markt und Wende ergeben einen Ausschlag bis etwa 80 (bzw. 20) – das reicht für normale
  Zwischentiefs und -hochs. Nur wenn zusätzlich die Bitcoin-Zyklus-Gruppe in dieselbe Richtung zeigt, kommen bis zu 15 Punkte
  dazu: Werte über 90 (bzw. unter 10) gibt es nur an großen Wendepunkten. Zeigt die Zyklus-Gruppe in die Gegenrichtung,
  dämpft sie leicht (höchstens 5 Punkte).</p>
  <p><b>Verbindung:</b> Lage und Markt bestimmen die <b>Richtung</b>, die Wende die <b>Bestätigung</b>. Fällt der Kurs noch
  („fallendes Messer“), wird das Signal auf ${p(Math.max(0.05, 2 * cfg.gb - 1))}–${p(cfg.gb)} gedämpft.
  Danach wird es mit dem <b>übergeordneten Zeitrahmen</b> (Stärke ${String(cfg.kh).replace('.', ',')}) und dem <b>Gesamtmarkt</b>
  (Top-20-Index, Stärke ${String(cfg.km).replace('.', ',')}) abgeglichen. Die Spreizung (Faktor ${cal.S.toFixed(2).replace('.', ',')})
  ist so gewählt, dass rund ${String(cfg.share * 100).replace('.', ',')} % der Kerzen in einer Zone liegen.</p>
  <p><b>Gesamt-Score:</b> unabhängig vom gewählten Chart. ${mode === 'long' ? '1 T (30 %), 1 W (45 %), 1 M (25 %)' : '1 Std (25 %), 4 Std (40 %), 1 T (35 %)'} –
  jeweils nur die zuletzt abgeschlossene Kerze, damit sich nichts nachträglich ändert. Weil das Mitteln den Wert Richtung 50 drückt, wird er wieder auf die übliche Schwankungsbreite gespreizt (Faktor ${String((D.overall_k || {})[mode] ?? 1).slice(0, 4).replace('.', ',')}). Im Score-Fenster kannst du zwischen
  „Zeitrahmen“ und „Gesamt“ wechseln; die Pfeile im Chart folgen dem angezeigten Score.</p>
  <p><b>Signale (einstellbar):</b> Kauf in ${nB} Tranche${nB > 1 ? 'n' : ''} bei Score ${buyLv().map(v => '≥ ' + String(+v.toFixed(1)).replace('.', ',')).join(', ')},
  Verkauf in ${nS} Tranche${nS > 1 ? 'n' : ''} bei ${sellLv().map(v => '≤ ' + String(+v.toFixed(1)).replace('.', ',')).join(', ')}. Jede Stufe zählt einmal
  je Runde; eine neue Runde beginnt erst, wenn der Score wieder ${D.reset} erreicht. Grün = kaufen, rot = verkaufen.</p>${calTxt}${evTxt}
  <p><b>Ehrlichkeit:</b> Jede Kerze nutzt nur Daten bis zu dieser Kerze – alte Signale ändern sich nicht. Die gestrichelten
  Hoch/Tief-Linien (Kursumkehr um mindestens die eingestellte Schwelle) und die ₿-Zyklus-Linien (Hoch gold, Tief türkis) (höchstes und tiefstes
  Bitcoin-Niveau je Halving-Zyklus, fest hinterlegt) stehen erst im Nachhinein fest und zeigen nur, wie gut der Score getroffen hätte. Kein Indikator kann
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
  'Score-Teil Lage': [40, 60], 'Score-Teil Wende': [40, 60], 'Score-Teil Markt': [40, 60], 'Score-Teil Zyklus': [40, 60],
  'BTC Mayer-Multiple': [0.8, 2.4], 'BTC Pi-Cycle': [0.45, 1], 'BTC 200-Wochen-Ø': [1, 4], 'Kreuz naht': [-0.3, 0.3],
  'Lage in 100-Kerzen-Spanne': [0.15, 0.85], 'Lage in 300-Kerzen-Spanne': [0.15, 0.85]};
function sheetList(){
  const names = Object.keys(D.info), grp = g => names.filter(k => D.go[k] === g);
  const box = k => `<label><input type="checkbox" value="${esc(k)}" ${panes.includes(k) ? 'checked' : ''}> ${esc(k)}</label>`;
  $('sheetl').innerHTML = `<p class="msg" style="text-align:left">Bis zu 4 Fenster. Jedes läuft synchron mit Chart, Zeitachse und Hoch/Tief-Linien.</p>` +
    `<h4>Score-Bestandteile</h4>${['Score-Teil Lage', 'Score-Teil Wende', 'Score-Teil Markt', 'Score-Teil Zyklus'].map(box).join('')}` +
    `<h4>Zyklen</h4>${box('Zyklen')}` +
    ['Lage', 'Wende', 'Markt', 'Zyklus'].map(g => `<h4>${g === 'Zyklus' ? 'Bitcoin-Zyklus' : g}</h4>` + grp(g).filter(k => k !== 'Zyklen').map(box).join('')).join('');
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

// ---------------- Optionen: Linien, Schwelle, Score-Art
const zArr = T => zsel === 'g' ? (T['zg' + sfx()] || T['z' + sfx()]) : (T['z' + sfx()] || null);
$('mas').innerHTML = MA.map(([k, l, c]) => `<button class="chip sm ma" data-k="${k}"><i style="background:${c}"></i>${l}</button>`).join('');
$('mas').addEventListener('click', e => { const b = e.target.closest('.ma'); if (!b) return; const k = b.dataset.k;
  mas = mas.includes(k) ? mas.filter(x => x !== k) : mas.concat(k); store('mas', JSON.stringify(mas)); syncOpts(); if (cur && cur.data) show(true); });
function maSeries(c, k){
  const n = c.length, out = new Array(n).fill(null);
  if (k === 'e20') { const a = 2 / 21; let e = null;
    for (let i = 0; i < n; i++) { e = e == null ? c[i][4] : a * c[i][4] + (1 - a) * e; if (i >= 19) out[i] = e; } return out; }
  const N = k === 's50' ? 50 : k === 's200' ? 200 : Math.round(200 * 604800 / SEC[ctf]);
  if (N < 2 || N > n) return null;
  let sum = 0; for (let i = 0; i < n; i++) { sum += c[i][4]; if (i >= N) sum -= c[i - N][4]; if (i >= N - 1) out[i] = sum / N; }
  return out;
}
function syncOpts(){
  document.querySelectorAll('#mas .ma').forEach(b => b.classList.toggle('on', mas.includes(b.dataset.k)));
  $('tgPiv').classList.toggle('on', showPiv); $('tgCyc').classList.toggle('on', showCyc);
  $('thrw').style.opacity = showPiv ? 1 : .35; $('thr').value = thrGet(); $('thrv').textContent = thrGet() + ' %';
  document.querySelectorAll('#zseg button').forEach(b => b.classList.toggle('on', b.dataset.z === zsel));
}
$('tgPiv').addEventListener('click', () => { showPiv = !showPiv; store('piv', showPiv ? '1' : '0'); syncOpts(); redraw(); });
$('tgCyc').addEventListener('click', () => { showCyc = !showCyc; store('cyc', showCyc ? '1' : '0'); syncOpts(); redraw(); });
$('thr').addEventListener('input', () => { store('thr_' + mode, $('thr').value); $('thrv').textContent = $('thr').value + ' %'; if (G) { G.zz = null; redraw(); } });
$('zseg').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; zsel = b.dataset.z; store('zsel', zsel); show(true); });
// Signale: Kauf-Tranche k beim ersten Eintritt über ihre Schwelle je Runde, Verkauf-Tranche k beim ersten Eintritt
// unter ihre Schwelle. Eine neue Runde beginnt, wenn der Score wieder die Mitte (50) erreicht.
function signalsJS(z, from){
  const out = [], f0 = from || 0;
  const run = (lv, up, code) => { let armed = true;
    for (let i = f0; i < z.length; i++) { const v = z[i]; if (v == null) continue; const p = i > f0 ? (z[i - 1] ?? 50) : 50;
      if (up ? v <= D.reset : v >= D.reset) armed = true;
      if (armed && (up ? v >= lv && p < lv : v <= lv && p > lv)) { out.push([i, code]); armed = false; } } };
  buyLv().forEach((lv, k) => run(lv, true, 'B' + (k + 1)));
  sellLv().forEach((lv, k) => run(lv, false, 'S' + (k + 1)));
  return out.sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : 1));
}
// Echte Hochs/Tiefs (im Nachhinein): Kursumkehr um mindestens die eingestellte Schwelle
function zigzag(c, th){
  const out = []; if (c.length < 3) return out;
  let dir = 0, hiP = c[0][2], hiI = 0, loP = c[0][3], loI = 0;
  for (let i = 1; i < c.length; i++) {
    const h = c[i][2], l = c[i][3];
    if (dir === 0) {
      if (h > hiP) { hiP = h; hiI = i; } if (l < loP) { loP = l; loI = i; }
      if (hiP >= loP * (1 + th) && hiI > loI) { out.push([loI, 'T', loP]); dir = 1; }
      else if (loP <= hiP * (1 - th) && loI > hiI) { out.push([hiI, 'H', hiP]); dir = -1; }
    } else if (dir === 1) {
      if (h > hiP) { hiP = h; hiI = i; }
      if (l <= hiP * (1 - th)) { out.push([hiI, 'H', hiP]); dir = -1; loP = l; loI = i; }
    } else {
      if (l < loP) { loP = l; loI = i; }
      if (h >= loP * (1 + th)) { out.push([loI, 'T', loP]); dir = 1; hiP = h; hiI = i; }
    }
  }
  return out;
}
function idxAt(t){ const c = G.T.c; let lo = 0, hi = c.length - 1; if (t < c[0][0]) return -1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (c[m][0] <= t) lo = m; else hi = m - 1; } return lo; }


// ---------------- Simulation (Backtest) mit deutschen Steuer-Varianten (Schätzung)
const SIMDEF = {start: '2022-01-01', budget: 10000, prod: 'spot', fee: 0.1, lev: 2, fund: 10, taxP: 35, freeP: 1000, taxC: 26.375, freeC: 1000};
let SIM = Object.assign({}, SIMDEF); try { Object.assign(SIM, JSON.parse(store('sim') || '{}')); } catch (e) {}
const SIMF = {simStart: 'start', simBudget: 'budget', simProd: 'prod', simFee: 'fee', simLev: 'lev', simFund: 'fund',
  simTaxP: 'taxP', simFreeP: 'freeP', simTaxC: 'taxC', simFreeC: 'freeC'};
Object.entries(SIMF).forEach(([id, k]) => { const el = $(id); el.value = SIM[k];
  el.addEventListener('change', () => { SIM[k] = el.type === 'number' ? +el.value : el.value; store('sim', JSON.stringify(SIM)); simulate(); }); });
function simulate(){
  if (!G) return;
  const T = G.T, c = T.c, Z = zArr(T), n = c.length, P = SIM;
  document.querySelectorAll('.futonly').forEach(e => e.style.display = P.prod === 'fut' ? '' : 'none');
  document.querySelectorAll('.spotonly').forEach(e => e.style.display = P.prod === 'spot' ? '' : 'none');
  document.querySelectorAll('.capOnly').forEach(e => e.style.display = P.prod === 'spot' ? 'none' : '');
  if (!Z) { $('simres').innerHTML = '<div class="cat">Kein Score für diesen Zeitrahmen – schalte auf „Gesamt“.</div>'; $('trades').innerHTML = ''; $('simnote').textContent = ''; return; }
  const t0 = Date.parse(P.start + 'T00:00:00Z') / 1000;
  let i0 = c.findIndex(k => k[0] >= t0); if (i0 < 0) i0 = n - 1;
  const fee = P.fee / 100, L = P.prod === 'fut' ? Math.max(1, P.lev) : 1, fund = P.prod === 'fut' ? P.fund / 100 : 0;
  const eurTo = t => ccy === 'EUR' ? 1 : (fxAt(t) || 1.1);        // Steuer-Freigrenzen sind in Euro
  const sig = signalsJS(Z, i0);
  let cash = P.budget, lots = [], rows = [], fees = 0, taxes = 0, buyPlan = 0, sellPlan = 0, si = 0;
  const yr = {};   // Gewinne je Jahr: priv (privates Veräußerungsgeschäft) / cap (Kapitalerträge)
  const addGain = (t, kind, g) => { const y = new Date(t * 1000).getUTCFullYear(); yr[y] = yr[y] || {priv: 0, cap: 0, t}; yr[y][kind] += g; yr[y].t = t; };
  const taxYear = y => { const g = yr[y]; if (!g || g.done) return 0; g.done = true; let tx = 0;
    if (g.priv > 0 && g.priv > P.freeP * eurTo(g.t)) tx += g.priv * P.taxP / 100;          // Freigrenze: darüber alles steuerpflichtig
    if (g.cap > 0) tx += Math.max(0, g.cap - P.freeC * eurTo(g.t)) * P.taxC / 100;        // Pauschbetrag: nur der Teil darüber
    return tx; };
  const lotValue = (lt, px, t) => { const fundCost = lt.margin * (lt.lev - 1 > 0 || P.prod === 'fut' ? lt.lev : 0) * fund * (t - lt.t) / 31557600;
    return P.prod === 'fut' ? Math.max(0, lt.margin + lt.units * (px - lt.entry) - fundCost) : lt.units * px; };
  let curYear = new Date(c[i0][0] * 1000).getUTCFullYear();
  for (let i = i0; i < n; i++) {
    const [t, , hi, lo, px] = c[i], y = new Date(t * 1000).getUTCFullYear();
    if (y !== curYear) { const tx = taxYear(curYear); if (tx) { cash -= tx; taxes += tx; rows.push([t, 'Steuer ' + curYear, null, tx, null, '']); } curYear = y; }
    if (P.prod === 'fut') lots = lots.filter(lt => { if (lotValue(lt, lo, t) > 0) return true;
      rows.push([t, 'Liquidation', lo, 0, -100, `Kauf ${fmtDate(lt.t)} @ ${price(lt.entry)}`]); addGain(t, 'cap', -lt.cost); return false; });
    while (si < sig.length && sig[si][0] === i) {
      const code = sig[si][1], k = +code.slice(1); si++;
      if (code[0] === 'B') {
        if (k === 1 || !buyPlan) buyPlan = cash / nB;
        const amt = Math.min(buyPlan, cash); if (amt < 1) continue;
        const f = amt * fee, inv = amt - f; fees += f; cash -= amt;
        lots.push({t, entry: px, units: inv * L / px, margin: inv, cost: amt, lev: L});
        rows.push([t, `Kauf ${nB > 1 ? k + '/' + nB : ''}`, px, amt, null, '']);
      } else {
        const total = lots.reduce((a, lt) => a + lt.units, 0); if (total <= 0) continue;
        if (k === 1 || !sellPlan) sellPlan = total / nS;
        let need = k === nS ? total : Math.min(sellPlan, total);
        while (need > 1e-12 && lots.length) {
          const lt = lots[0], u = Math.min(lt.units, need), fr = u / lt.units;
          const val = lotValue(lt, px, t) * fr, f = u * px * fee, net = Math.max(0, val - f), basis = lt.cost * fr;
          const gain = net - basis, held = (t - lt.t) / 86400;
          fees += f; cash += net;
          let kind = P.prod === 'spot' ? (held > 365 ? null : 'priv') : 'cap';
          if (kind) addGain(t, kind, gain);
          rows.push([t, `Verk. ${nS > 1 ? k + '/' + nS : ''}`, px, net, basis ? (net / basis - 1) * 100 : 0,
            `Kauf ${fmtDate(lt.t)} @ ${price(lt.entry)} · ${Math.round(held)} T · ${kind ? 'steuerpflichtig' : 'steuerfrei (> 1 Jahr)'}`]);
          lt.units -= u; lt.margin *= (1 - fr); lt.cost -= basis; need -= u;
          if (lt.units <= 1e-12) lots.shift();
        }
      }
    }
  }
  const tN = c[n - 1][0], pN = c[n - 1][4];
  const due = taxYear(curYear); cash -= due; taxes += due;            // Steuer des laufenden Jahres (bereits realisiert)
  const open = lots.reduce((a, lt) => a + lotValue(lt, pN, tN), 0);
  // Steuer, falls die offenen Positionen heute verkauft würden (Schätzung)
  let taxOpen = 0;
  lots.forEach(lt => { const g = lotValue(lt, pN, tN) - lt.cost, held = (tN - lt.t) / 86400;
    if (g > 0) taxOpen += P.prod === 'spot' ? (held > 365 ? 0 : g * P.taxP / 100) : g * P.taxC / 100; });
  const value = cash + open, after = value - taxOpen;
  // Vergleich: am Startdatum alles kaufen und halten (ohne Hebel)
  const p0 = c[i0][4], hold = P.budget * (1 - fee) / p0 * pN * (1 - fee), hg = hold - P.budget, heldD = (tN - c[i0][0]) / 86400;
  const holdTax = hg > 0 ? (P.prod === 'spot' ? (heldD > 365 ? 0 : (hg > P.freeP * eurTo(tN) ? hg * P.taxP / 100 : 0)) : Math.max(0, hg - P.freeC * eurTo(tN)) * P.taxC / 100) : 0;
  const card = (l, v, sub, col) => `<div class="cat">${l}<b style="color:${col || 'var(--txt)'}">${v}</b>${sub || ''}</div>`;
  const gp = (after / P.budget - 1) * 100, hp = ((hold - holdTax) / P.budget - 1) * 100;
  $('simres').innerHTML =
    card('Wert heute (nach Steuer)', money(after), pct(gp), gp >= 0 ? 'var(--buy)' : 'var(--sell)') +
    card('Kaufen & Halten', money(hold - holdTax), pct(hp), hp >= 0 ? 'var(--buy)' : 'var(--sell)') +
    card('Davon offen investiert', money(open), lots.length ? `${lots.length} Position${lots.length > 1 ? 'en' : ''}` : 'alles verkauft') +
    card('Steuern', money(taxes + taxOpen), taxOpen ? `inkl. ${money(taxOpen)} bei Verkauf heute` : 'bezahlt') +
    card('Gebühren', money(fees), `${rows.filter(r => r[2] != null).length} Trades`) +
    card('Zeitraum', `${fmtDate(c[i0][0])}`, `bis ${fmtDate(tN)} · ${TFL[ctf]}`);
  const prodTxt = {spot: `Spot: Gewinne nach über 1 Jahr Haltedauer steuerfrei, sonst ${String(P.taxP).replace('.', ',')} % (Freigrenze ${P.freeP} € pro Jahr), Verkauf nach First in, First out.`,
    etp: `ETP: Abgeltungsteuer ${String(P.taxC).replace('.', ',')} % auf Gewinne über ${P.freeC} € pro Jahr, keine Haltefrist.`,
    fut: `Futures mit Hebel ${L}×: Abgeltungsteuer ${String(P.taxC).replace('.', ',')} % über ${P.freeC} €, Finanzierung ${String(P.fund).replace('.', ',')} % p. a. auf den Hebelbetrag, Liquidation bei Totalverlust der Position. Verlustverrechnung vereinfacht.`};
  $('simnote').innerHTML = `${prodTxt[P.prod]} Ausführung zum Schlusskurs der Signal-Kerze, ${nB} Kauf- und ${nS} Verkaufs-Tranche${nS > 1 ? 'n' : ''}.
    Steuern nur grob geschätzt – keine Steuerberatung. Die Einstellungen des Scores wurden mit Wissen über die Vergangenheit kalibriert, daher wirkt der Backtest eher zu gut.`;
  const show_ = rows.slice().reverse().slice(0, 300);
  $('trades').innerHTML = show_.length ? `<tr><th>Datum</th><th>Aktion</th><th>Preis</th><th>Betrag</th><th>Ergebnis</th></tr>` + show_.map(r =>
    `<tr><td>${fmtDate(r[0])}</td><td class="${r[1].startsWith('Kauf') ? 'up' : 'dn'}">${esc(r[1])}</td><td>${r[2] == null ? '' : price(r[2])}</td>
     <td>${money(r[3])}</td><td>${r[4] == null ? '' : `<b class="${r[4] >= 0 ? 'up' : 'dn'}">${pct(r[4])}</b>`}<br><small>${esc(r[5])}</small></td></tr>`).join('')
    : '<tr><td>Keine Signale im gewählten Zeitraum.</td></tr>';
}

// ---------------- Detail
function destroy(){ if (G) G.charts.forEach(c => c.chart.remove()); G = null; }
function closeCoin(){ if (desktop() && D.coins.length) { openCoin(D.coins[0].s); return; }
  destroy(); cur = null; $('detail').classList.remove('open'); document.body.style.overflow = ''; }
$('tfc').innerHTML = D.tfs.map(([k, l]) => `<button class="chip" data-t="${k}">${l}</button>`).join('');
$('tfc').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b && !b.disabled) { ctf = b.dataset.t; store('ctf', ctf); view = null; M = null; show(); } });
function header(){
  const meta = cur.meta;
  $('dt').textContent = landscape() ? meta.s : meta.s + ' · ' + meta.n;
  $('ds').innerHTML = price(toC(meta.p)) + ' <span class="' + (meta.ch >= 0 ? 'up' : 'dn') + '">' + pct(meta.ch) + '</span> · #' + meta.r;
}
async function openCoin(sym){
  const meta = D.coins.find(x => x.s === sym); if (!meta) return closeCoin();
  destroy(); view = null; M = null;
  $('detail').classList.add('open'); $('dmain').scrollTop = 0; $('detail').scrollTop = 0;
  document.body.style.overflow = desktop() ? '' : 'hidden';
  document.querySelectorAll('li.row').forEach(li => li.classList.toggle('sel', li.dataset.s === sym));
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
function convT(T){
  if (ccy !== 'EUR' || !FX.length) return T;
  if (!T._eur || T._eurN !== T.c.length || T._eurF !== T.c[0][0]) {
    T._eur = T.c.map(k => { const f = fxAt(k[0]) || 1; return [k[0], k[1] / f, k[2] / f, k[3] / f, k[4] / f]; });
    T._eurN = T.c.length; T._eurF = T.c[0][0];
  }
  return Object.assign({}, T, {c: T._eur});
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
  const T = tfs[ctf], Z = zArr(T), hasZ = !!Z, ownZ = !!T['z' + sfx()];
  const s = mode === 'long' ? cur.meta.gl : cur.meta.g, col = color(s);
  $('dsc').innerHTML = `<small style="display:block;font-size:9px;font-weight:600;opacity:.8">Gesamt</small>${s == null ? '–' : Math.round(s)}`;
  $('dsc').style.color = col; $('dsc').style.background = col + '22';
  ['fbox', 'catbox', 'pcbox', 'allbox', 'statbox'].forEach(id => $(id).style.display = ownZ && T.b ? '' : 'none');
  $('noz').style.display = hasZ ? 'none' : '';
  syncOpts();
  if (ownZ && T.b) details(T);
  if (!keep) M = null;
  const ind = await loadInd(T);
  const Tc = convT(T);
  destroy();
  waitLib(() => { draw(Tc, ind); simulate(); });
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
  $('cats').innerHTML = ['Lage', 'Markt', 'Wende', 'Zyklus'].map(g => {
    const v = k[g]; if (v == null) return `<div class="cat">${g}<b>–</b></div>`;
    return `<div class="cat">${g}<b style="color:${color(v)}">${v}</b><div class="bar"><i style="width:${v}%;background:${color(v)}"></i></div></div>`;
  }).join('');
  const sorted = [...T.b].sort((a, b) => b[2] - a[2]);
  const item = x => `<div class="it"><b><span style="color:var(--txt)">${esc(x[0])}</span><span style="color:${color(x[2])}">${x[2]}</span></b><span>${esc(x[4])}</span></div>`;
  const pro = sorted.filter(x => x[2] >= 60).slice(0, 4), con = sorted.filter(x => x[2] <= 40).reverse().slice(0, 4);
  $('pro').innerHTML = pro.length ? pro.map(item).join('') : '<div class="it"><span>Keine deutlichen Kaufsignale</span></div>';
  $('con').innerHTML = con.length ? con.map(item).join('') : '<div class="it"><span>Keine deutlichen Verkaufssignale</span></div>';
  $('all').innerHTML = ['Lage', 'Markt', 'Wende', 'Zyklus'].map(g => `<tr><td colspan="3" style="color:var(--mut);font-weight:600">${g}</td></tr>` +
    T.b.filter(x => x[1] === g).map(x => `<tr><td>${esc(x[0])}<br><span style="color:var(--mut);font-size:11px">${esc(x[4])}</span></td>
     <td class="v">${esc(x[3])}</td><td class="v"><span class="mini"><i style="width:${x[2]}%;background:${color(x[2])}"></i></span> <b style="color:${color(x[2])}">${x[2]}</b></td></tr>`).join('')).join('');
}

// ---------------- Charts: Kurs, Score und gewählte Indikatoren untereinander
function heights(k, hasZ){
  if (desktop()) {
    const zh = hasZ ? 170 : 0, ph = 120;
    return {price: Math.max(340, window.innerHeight - 300 - zh - ph * k), score: zh, pane: ph};
  }
  if (landscape()) {
    const box = $('cwrap').getBoundingClientRect(), top = Math.max(0, box.top - $('dmain').getBoundingClientRect().top);
    const total = Math.max(220, window.innerHeight - top - AXH() - 4);
    const zh = hasZ ? Math.max(56, Math.round(total * 0.2)) : 0, ph = k ? Math.max(46, Math.round(total * 0.13)) : 0;
    return {price: Math.max(110, total - zh - ph * k), score: zh, pane: ph};
  }
  return {price: 320, score: hasZ ? 150 : 0, pane: 105};
}
function draw(T, ind){
  const L = window.LightweightCharts, Z = zArr(T), hasZ = !!Z, pre = T.pre || 0;
  const wrap = $('cwrap');
  const list = [{kind: 'price'}];
  if (hasZ) list.push({kind: 'score'});
  if (ind && ind.v) panes.forEach(name => { if (ind.v[name]) list.push({kind: 'ind', name}); });
  const H = heights(list.length - 1 - (hasZ ? 1 : 0), hasZ);
  wrap.innerHTML = list.map((p, i) => {
    const h = (p.kind === 'price' ? H.price : p.kind === 'score' ? H.score : H.pane) + (i === list.length - 1 ? AXH() : 0);
    const lab = p.kind === 'ind' ? `<div class="plab" id="plab${i}">${esc(p.name)}</div>` : p.kind === 'score' ? `<div class="plab" id="plab${i}"></div>` : '';
    const bot = i === list.length - 1 ? AXH() : 0;
    return `<div class="cw" style="height:${h}px"><div class="ch" id="ch${i}" style="height:${h}px"></div>${lab}
      ${p.kind === 'price' ? '<div class="layer" id="ml"></div>' : ''}
      <div class="touch${i ? ' pany' : ''}" data-i="${i}" style="bottom:${bot}px;right:78px"></div>
      <div class="pscroll" style="bottom:${bot}px"></div></div>`;
  }).join('') + `<div class="layer" id="pvl" style="bottom:${AXH()}px"></div>
    <div class="axt" id="axt" style="height:${AXH()}px"><span>${desktop() ? 'Zeitachse: ziehen = schieben · Mausrad = zoomen' : 'Zeitachse: ziehen = schieben · 2 Finger = zoomen'}</span></div>`;
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
      const sg = hasZ ? signalsJS(Z) : [], mk = [];
      sg.forEach(([j, code]) => { if (j < 0 || j >= n) return;
        const buy = code[0] === 'B', k = +code.slice(1), N = buy ? nB : nS, pr = price(T.c[j][4]);
        mk.push({time: T.c[j][0], position: buy ? 'belowBar' : 'aboveBar', color: buy ? '#22c55e' : '#ef4444',
          shape: buy ? 'arrowUp' : 'arrowDown', size: N > 1 ? 0.9 : 1.2,
          text: (buy ? 'Kauf' : 'Verk.') + (N > 1 ? ` ${k}/${N}` : '') + ' ' + pr}); });
      mk.sort((a, b) => a.time - b.time);
      series.setMarkers(mk);
      MA.forEach(([k, l, col]) => { if (!mas.includes(k)) return; const v = maSeries(T.c, k); if (!v) return;
        const ls = chart.addLineSeries({color: col, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false,
          crosshairMarkerVisible: false, priceFormat: {type: 'price', precision: prec, minMove: Math.pow(10, -prec)}});
        ls.setData(T.c.map((c, j) => v[j] == null ? {time: c[0]} : {time: c[0], value: v[j]})); });
    } else if (p.kind === 'score') {
      vals = Z;
      series = chart.addBaselineSeries({baseValue: {type: 'price', price: 50}, lineWidth: 2,
        topLineColor: '#22c55e', topFillColor1: 'rgba(34,197,94,.28)', topFillColor2: 'rgba(34,197,94,.02)',
        bottomLineColor: '#ef4444', bottomFillColor1: 'rgba(239,68,68,.02)', bottomFillColor2: 'rgba(239,68,68,.28)',
        priceFormat: {type: 'price', precision: 0, minMove: 1}, lastValueVisible: true, priceLineVisible: false,
        autoscaleInfoProvider: () => ({priceRange: {minValue: 0, maxValue: 100}})});
      series.setData(T.c.map((k, j) => Z[j] == null ? {time: k[0]} : {time: k[0], value: Z[j]}));
      buyLv().forEach((lv, q) => series.createPriceLine({price: lv, color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: q === 0, title: q === 0 ? 'Kauf' : ''}));
      sellLv().forEach((lv, q) => series.createPriceLine({price: lv, color: '#ef4444', lineWidth: 1, lineStyle: 2, axisLabelVisible: q === 0, title: q === 0 ? 'Verk.' : ''}));
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
      document.querySelectorAll('#cwrap .touch').forEach(el => el.style.right = w + 'px');
      document.querySelectorAll('#cwrap .pscroll').forEach(el => el.style.width = w + 'px');
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
    el.innerHTML = (c.kind === 'score' ? `${zsel === 'g' ? 'Gesamt-Score' : 'Score ' + TFL[ctf]} · ${esc(D.modes[mode].label)}` : esc(c.name)) + ` <b>${c.kind === 'score' ? (v == null ? '–' : Math.round(v)) : num(v)}</b>`; });
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

// ---------------- Hoch/Tief-Linien mit Preis + goldene Bitcoin-Zyklus-Linien
function drawPiv(){
  const el = $('pvl'); if (!el) return; el.innerHTML = '';
  if (!G) return;
  const ts = G.pc.timeScale(), out = [], W = $('cwrap').clientWidth;
  if (showPiv) {
    if (!G.zz) G.zz = zigzag(G.T.c, thrGet() / 100);
    $('thrv').textContent = thrGet() + ' % · ' + G.zz.length;
    let lastX = {H: -1e9, T: -1e9};
    for (const [i, k, p] of G.zz) {
      const x = ts.logicalToCoordinate(i); if (x == null || x < -2 || x > W) continue;
      const col = k === 'H' ? 'rgba(239,68,68,.65)' : 'rgba(34,197,94,.65)';
      out.push(`<div class="vl" style="left:${x}px;border-color:${col}"></div>`);
      const y = G.cs.priceToCoordinate(p);
      if (y != null && x - lastX[k] > 58) {
        lastX[k] = x;
        out.push(`<div class="ptag ${k}" style="left:${x}px;top:${Math.max(showCyc && D.btcc ? 62 : 0, k === 'H' ? y - 18 : y + 4)}px">${price(p)}</div>`);
      }
    }
  }
  if (showCyc && D.btcc) {
    const vis = [];
    for (const [t, k, p, ok] of D.btcc) {
      const i = idxAt(t); if (i < 0) continue;
      const x = ts.logicalToCoordinate(i); if (x == null || x < -2 || x > W) continue;
      vis.push({x, t, k, p, ok});
    }
    // Schilder versetzt in Reihen, damit sie sich nie überlappen; wird es zu eng, nur Kurzform
    const ROWS = 3, rowEnd = new Array(ROWS).fill(-1e9);
    const crowded = vis.length > 1 && Math.min(...vis.slice(1).map((v, j) => v.x - vis[j].x)) < 70;
    for (const v of vis) {
      const d = new Date(v.t * 1000), mm = `${d.getMonth() + 1}/${String(d.getFullYear()).slice(crowded ? 2 : 0)}`;
      const txt = crowded ? `${v.k === 'H' ? '▲' : '▼'} ${mm}` : `₿-Zyklus-${v.k === 'H' ? 'Hoch' : 'Tief'} ${mm}<br>${price(toC(v.p, v.t))}`;
      const w = crowded ? 54 : 118, left = Math.min(v.x + 4, W - w - 2);
      let row = rowEnd.findIndex(e => e < left - 4);
      out.push(`<div class="vl cyc ${v.k}" style="left:${v.x}px"></div>`);
      if (row < 0) continue;
      rowEnd[row] = left + w;
      out.push(`<div class="ctag ${v.k}" style="left:${left}px;top:${2 + row * (crowded ? 18 : 30)}px;width:${w}px" title="${price(v.p)}">${txt}</div>`);
    }
  }
  el.innerHTML = out.join('');
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
    const H0 = $('ch0').clientHeight;
    const tag = (q, P, c2, role) => { if (P.x == null || P.y == null) return '';
      const above = P.y > 60, left = Math.min(Math.max(P.x, 64), W - 64), act = M.drag === role ? ' grab' : '';
      return `<div class="bub sm${act}" data-role="${role}" style="left:${left}px;top:${above ? P.y - 10 : P.y + 54}px;border-color:${c2}"><b>${price(q.p)}</b><br>${fmtDate(timeOf(q.l))}<i>✥</i></div>`; };
    html += tag(M.a, pa, '#60a5fa', 'a') + tag(B, pb, col, 'b');
    if (pa.x != null && pb.x != null && pa.y != null && pb.y != null) {
      const cx = Math.min(Math.max((pa.x + pb.x) / 2, 70), W - 70), cy = Math.min(Math.max((pa.y + pb.y) / 2 + 14, 30), H0 - 10);
      html += `<div class="bub mid" style="left:${cx}px;top:${cy}px;border-color:${col}"><b style="color:${col}">${pct(pc_)}</b><br>${d}</div>`;
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
        // Offene Messung: Blase (oder Punkt) greifen = diesen Messpunkt nachjustieren, der andere bleibt stehen
        if (fingers.size === 0 && M && M.moved) {
          const hit = [...document.querySelectorAll('#ml .bub[data-role]')].find(b => { const r = b.getBoundingClientRect();
            return e.clientX >= r.left - 10 && e.clientX <= r.right + 10 && e.clientY >= r.top - 10 && e.clientY <= r.bottom + 10; });
          const ts = G.pc.timeScale(), r0 = $('ch0').getBoundingClientRect();
          const near = ['a', 'b'].find(k => { const x = ts.logicalToCoordinate(M[k].l), y = G.cs.priceToCoordinate(M[k].p);
            return x != null && y != null && Math.hypot(e.clientX - r0.left - x, e.clientY - r0.top - y) < 26; });
          const role = hit ? hit.dataset.role : near;
          if (role) {
            const x = ts.logicalToCoordinate(M[role].l), y = G.cs.priceToCoordinate(M[role].p);
            M.off = {x: r0.left + x - e.clientX, y: r0.top + y - e.clientY};   // Punkt springt nicht unter den Finger
            M.drag = role; fingers.set(e.pointerId, role); drawM(); return;
          }
        }
        if (fingers.size === 0 || !M) { fingers.clear(); fingers.set(e.pointerId, 'b'); M = {a: q, b: q, moved: false, x0: e.clientX, y0: e.clientY}; }
        else if (fingers.size === 1) { fingers.set(e.pointerId, 'a'); M.a = q; M.moved = true; }
        drawM();
      });
      el.addEventListener('pointermove', e => {
        if (!G) return;
        const role = fingers.get(e.pointerId);
        const q = M && M.drag && role === M.drag && M.off ? pointAt({clientX: e.clientX + M.off.x, clientY: e.clientY + M.off.y}) : pointAt(e);
        if (!q) return;
        if (!role) { if (e.pointerType === 'mouse' && fingers.size === 0) { M = {a: q, b: q, moved: false, hover: true}; drawM(); } return; }
        M[role] = q;
        if (!M.moved && Math.hypot(e.clientX - M.x0, e.clientY - M.y0) > 8) M.moved = true;
        drawM();
      });
      ['pointerup', 'pointercancel'].forEach(ev => el.addEventListener(ev, e => { fingers.delete(e.pointerId);
        if (M && M.drag && fingers.size === 0) { M.drag = null; M.off = null; drawM(); } }));
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
      el.addEventListener('pointerdown', e => mv(e));
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
setLtf(ltf === 'g' || D.ltfs.includes(ltf) ? ltf : 'g');
explain();
const h0 = decodeURIComponent(location.hash.slice(1));
if (h0) openCoin(h0); else if (desktop() && D.coins.length) openCoin(D.coins[0].s);
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
