#!/usr/bin/env python3
"""
Krypto-Dashboard v2 – läuft automatisch über GitHub Actions.

1. Holt die Top-Coins nach Marktkapitalisierung (CoinGecko, Ersatz: CoinPaprika),
   ohne Stablecoins und Wrapped/Staked-Kopien.
2. Lädt pro Coin Kerzen in 9 Zeitrahmen: 1 Std, 4 Std, 1 Tag, 5 Tage, 1 Woche,
   2 Wochen, 1 Monat, 6 Monate, 1 Jahr (Binance, Ersatz: OKX, KuCoin – ohne API-Key).
3. Berechnet einen antizyklischen Score 0–100:
   hoch = Kurs weit unten + erste Wende-Zeichen (Kaufzone),
   tief = Kurs weit oben + erste Schwäche-Zeichen (Verkaufszone).
   Der Score nutzt nur Daten bis zur jeweiligen Kerze (kein Repainting).
4. Markiert zum Vergleich die echten Hochs und Tiefs (die erst im Nachhinein feststehen).
5. Schreibt site/index.html (Liste) und site/data/<COIN>.json (Charts).

Keine Anlageberatung.
"""
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
DISPLAY = 300          # max. Kerzen pro Zeitrahmen im Chart
SCORE_MIN = 80         # so viele Kerzen braucht es mindestens für einen Score
BUY_LEVEL = 80
SELL_LEVEL = 20
STRETCH = 1.75         # Spreizung des Scores um 50
PIVOT_N = 10           # echtes Hoch/Tief = höchster/tiefster Punkt ±10 Kerzen
STAT_HORIZON = 10      # Trefferquote: Kursänderung 10 Kerzen nach dem Signal
OUT_DIR = "site"

LIST_TFS = ["1h", "4h", "1d", "1w"]   # Scores, die in der Liste wählbar sind
TIMEFRAMES = [("1h", "1 Std"), ("4h", "4 Std"), ("1d", "1 T"), ("5d", "5 T"), ("1w", "1 W"),
              ("2w", "2 W"), ("1M", "1 M"), ("6M", "6 M"), ("1y", "1 J")]
NATIVE = {"1h": (3600, 600), "4h": (14400, 600), "1d": (86400, 1000),
          "1w": (604800, 1000), "1M": (2678400, 1000)}   # Sekunden, Anzahl

GROUP_WEIGHTS = {"Lage": 0.65, "Wende": 0.35}
# Stärkeres Gewicht für Indikatoren, die die echte Kurslage messen (nicht nur das Tempo)
IND_WEIGHTS = {"Lage in 100-Kerzen-Spanne": 2.5, "Abstand EMA 50": 1.5, "Abstand SMA 200": 1.5,
               "Bollinger %B": 1.0, "RSI (14)": 0.75, "Stochastic RSI": 0.5, "Williams %R": 0.5,
               "CCI (20)": 0.5, "MFI (14)": 0.75,
               "Erholung vom Tief": 2.0, "Stoch-RSI Kreuzung": 1.0, "RSI-Schwung (3)": 1.0,
               "MACD dreht": 1.0, "RSI-Divergenz": 1.0, "Volumen-Klimax": 0.5, "Kerzen-Docht": 0.5}
IND_INFO = {
    "RSI (14)": "Relative Stärke der letzten 14 Kerzen. Unter 30 = überverkauft (Kauf), über 70 = überkauft (Verkauf).",
    "Stochastic RSI": "Wo steht der RSI innerhalb seiner eigenen Spanne? Ganz unten = Kauf, ganz oben = Verkauf.",
    "Williams %R": "Schlusskurs im Verhältnis zur 14-Kerzen-Spanne. Nahe Tief = Kauf, nahe Hoch = Verkauf.",
    "CCI (20)": "Abweichung vom Durchschnittspreis. Stark negativ = Kauf, stark positiv = Verkauf.",
    "MFI (14)": "RSI mit Volumen. Niedrig = Geld ist abgeflossen (Kauf), hoch = überhitzt (Verkauf).",
    "Bollinger %B": "Lage zwischen den Bollinger-Bändern. Am unteren Band = Kauf, am oberen = Verkauf.",
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
}
GATE_BASE, GATE_GAIN, BOUNCE_DIV = 0.4, 0.6, 2.0

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
SESSION.headers["User-Agent"] = "krypto-dashboard/2.0 (+github-actions)"
SESSION.headers["Accept"] = "application/json"


class Blocked(Exception):
    pass


class NotFound(Exception):
    pass


def log(*a):
    print(*a, flush=True)


def get_json(url, params=None, tries=4, timeout=20):
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
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
            return r.json()
        except (Blocked, NotFound):
            raise
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


# ----------------------------------------------------------------- Coin-Liste
def coins_from_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    rows = get_json(url, {"vs_currency": "usd", "order": "market_cap_desc", "per_page": CANDIDATES,
                          "page": 1, "sparkline": "false", "price_change_percentage": "24h"})
    stable_ids = set()
    try:
        time.sleep(3)
        st = get_json(url, {"vs_currency": "usd", "category": "stablecoins", "per_page": 250, "page": 1})
        stable_ids = {c["id"] for c in st}
    except Exception as e:
        log("  Stablecoin-Kategorie nicht abrufbar:", e)
    return [{
        "sym": (c.get("symbol") or "").upper(), "name": c.get("name") or "",
        "rank": c.get("market_cap_rank") or 9999, "price": c.get("current_price"),
        "mcap": c.get("market_cap"), "chg": c.get("price_change_percentage_24h"),
        "stable": c.get("id") in stable_ids,
    } for c in rows]


def coins_from_coinpaprika():
    rows = get_json("https://api.coinpaprika.com/v1/tickers", {"quotes": "USD"}, timeout=40)
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
        data = get_json(f"{base}/api/v3/klines",
                        {"symbol": f"{sym}USDT", "interval": tf, "limit": NATIVE[tf][1]})
        return to_df([(int(k[0]) // 1000, k[1], k[2], k[3], k[4], k[5]) for k in data])
    return fetch


OKX_BAR = {"1h": "1Hutc", "4h": "4Hutc", "1d": "1Dutc", "1w": "1Wutc", "1M": "1Mutc"}


def okx(sym, tf):
    data = get_json("https://www.okx.com/api/v5/market/candles",
                    {"instId": f"{sym}-USDT", "bar": OKX_BAR[tf], "limit": 300})
    if str(data.get("code")) != "0" or not data.get("data"):
        raise NotFound(data.get("msg") or "leer")
    return to_df([(int(k[0]) // 1000, k[1], k[2], k[3], k[4], k[5]) for k in data["data"]])


KUCOIN_TYPE = {"1h": "1hour", "4h": "4hour", "1d": "1day", "1w": "1week"}


def kucoin(sym, tf):
    if tf == "1M":  # KuCoin hat keine Monatskerzen -> aus Tageskerzen bauen
        return aggregate(kucoin(sym, "1d"), "month")
    end = int(time.time())
    data = get_json("https://api.kucoin.com/api/v1/market/candles",
                    {"type": KUCOIN_TYPE[tf], "symbol": f"{sym}-USDT",
                     "startAt": end - min(NATIVE[tf][1], 1500) * NATIVE[tf][0], "endAt": end})
    if str(data.get("code")) != "200000" or not data.get("data"):
        raise NotFound(data.get("msg") or "leer")
    # KuCoin: Zeit, Open, Close, High, Low, Volumen
    return to_df([(int(k[0]), k[1], k[3], k[4], k[2], k[5]) for k in data["data"]])


SOURCES = [("Binance", binance("https://data-api.binance.vision")),
           ("Binance", binance("https://api.binance.com")),
           ("OKX", okx), ("KuCoin", kucoin)]
DEAD = set()


def aggregate(df, how):
    """Fasst Kerzen zu größeren Zeitrahmen zusammen."""
    if df is None or df.empty:
        return df
    dt = pd.to_datetime(df["t"], unit="s", utc=True)
    if how == "5d":
        key = df["t"] // (5 * 86400)
    elif how == "2w":
        key = (df["t"] - 4 * 86400) // (14 * 86400)   # Wochen starten montags
    elif how == "month":
        key = dt.dt.year * 12 + dt.dt.month
    elif how == "6M":
        key = dt.dt.year * 2 + (dt.dt.month - 1) // 6
    else:  # Jahr
        key = dt.dt.year
    g = df.groupby(key.values, sort=True)
    out = pd.DataFrame({"t": g["t"].first(), "o": g["o"].first(), "h": g["h"].max(),
                        "l": g["l"].min(), "c": g["c"].last(), "v": g["v"].sum()}).reset_index(drop=True)
    if how in ("5d", "2w") and len(out) > 3:
        out = out.iloc[1:].reset_index(drop=True)  # erste Gruppe ist evtl. unvollständig
    return out


def load_all(sym):
    """Findet eine Börse mit 4h-Kerzen und lädt dort alle Zeitrahmen."""
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
                time.sleep(0.12)
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
        return name, frames
    return None, None


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


def fmt(x, d=1):
    try:
        return "–" if x is None or math.isnan(x) else f"{x:.{d}f}"
    except TypeError:
        return "–"


def analyse(df):
    """Antizyklischer Score. Gibt (Score-Serie, Details letzte Kerze, Gruppen-Scores) zurück."""
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
    mad = tp.rolling(20).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
    cci = (tp - sma(tp, 20)) / (0.015 * mad.replace(0, np.nan))
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    mfi = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).fillna(100).where(pos.notna())
    bb_std = c.rolling(20).std(ddof=0)
    pctb = (c - (sma(c, 20) - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    e50 = ema(c, 50)
    s200 = sma(c, 200)
    z50 = (c - e50) / atr
    z200 = (c - s200) / atr
    hh100 = h.rolling(100, min_periods=50).max()
    ll100 = l.rolling(100, min_periods=50).min()
    rpos = (c - ll100) / (hh100 - ll100).replace(0, np.nan)

    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    hist_turn = (hist - hist.shift(2)) / (0.15 * atr)
    # RSI-Divergenz: neues Tief beim Kurs, aber höheres Tief beim RSI (und umgekehrt)
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

    # (Name, Gruppe, Teil-Score, Wert-Text, Erklärung)
    ind = [
        ("RSI (14)", "Lage", lin(rsi, 25, 75), rsi.map(fmt),
         rsi.map(lambda x: "überverkauft" if x < 30 else "überkauft" if x > 70 else "neutral")),
        ("Stochastic RSI", "Lage", lin(k, 10, 90), k.map(fmt),
         k.map(lambda x: "ganz unten" if x < 15 else "ganz oben" if x > 85 else "Mitte")),
        ("Williams %R", "Lage", lin(wr, -90, -10), wr.map(fmt),
         wr.map(lambda x: "überverkauft" if x < -80 else "überkauft" if x > -20 else "neutral")),
        ("CCI (20)", "Lage", lin(cci, -200, 200), cci.map(lambda x: fmt(x, 0)),
         cci.map(lambda x: "weit unter Schnitt" if x < -100 else "weit über Schnitt" if x > 100 else "normal")),
        ("MFI (14)", "Lage", lin(mfi, 15, 85), mfi.map(fmt),
         mfi.map(lambda x: "Geld fließt stark ab" if x < 20 else "überhitzt" if x > 80 else "neutral")),
        ("Bollinger %B", "Lage", lin(pctb, 0, 1), pctb.map(lambda x: fmt(x, 2)),
         pctb.map(lambda x: "unteres Band" if x < 0.1 else "oberes Band" if x > 0.9 else "innerhalb")),
        ("Abstand EMA 50", "Lage", lin(z50, -5, 5), z50.map(lambda x: fmt(x, 1) + " ATR"),
         z50.map(lambda x: "weit darunter" if x < -2.5 else "weit darüber" if x > 2.5 else "nahe dran")),
        ("Abstand SMA 200", "Lage", lin(z200, -12, 12), z200.map(lambda x: fmt(x, 1) + " ATR"),
         z200.map(lambda x: "–" if np.isnan(x) else "deutlich darunter" if x < -5 else "deutlich darüber" if x > 5 else "nahe dran")),
        ("Lage in 100-Kerzen-Spanne", "Lage", lin(rpos, 0, 1), rpos.map(lambda x: fmt(x * 100, 0) + " %"),
         rpos.map(lambda x: "nahe Tief" if x < 0.15 else "nahe Hoch" if x > 0.85 else "Mitte")),
        ("MACD dreht", "Wende", tanh_score(hist_turn), hist.map(lambda x: f"{x:.3g}"),
         hist_turn.map(lambda x: "dreht nach oben" if x > 0.3 else "dreht nach unten" if x < -0.3 else "flach")),
        ("RSI-Divergenz", "Wende", div, (r_lo_now - r_lo_prev).map(lambda x: fmt(x)),
         div.map(lambda x: "bullische Divergenz" if x > 60 else "bärische Divergenz" if x < 40 else "keine")),
        ("Stoch-RSI Kreuzung", "Wende", tanh_score((k - d) / 10), (k - d).map(fmt),
         (k - d).map(lambda x: "K kreuzt hoch" if x > 2 else "K kreuzt runter" if x < -2 else "neutral")),
        ("RSI-Schwung (3)", "Wende", tanh_score(rsi_mom / 8), rsi_mom.map(lambda x: fmt(x)),
         rsi_mom.map(lambda x: "RSI steigt" if x > 2 else "RSI fällt" if x < -2 else "seitwärts")),
        ("Volumen-Klimax", "Wende", climax, vr.map(lambda x: fmt(x, 1) + "×"),
         pd.Series(["Ausverkauf mit Volumen" if r > 1.5 and b < -0.3 else "Kaufrausch mit Volumen"
                    if r > 1.5 and b > 0.3 else "normal" for r, b in zip(vr.fillna(0), body)])),
        ("Erholung vom Tief", "Wende", tanh_score(bounce / BOUNCE_DIV), bounce.map(lambda x: fmt(x, 1) + " ATR"),
         bounce.map(lambda x: "hebt vom Tief ab" if x > 1 else "rutscht vom Hoch ab" if x < -1 else "unentschieden")),
        ("Kerzen-Docht", "Wende", (50 + 50 * wick).clip(0, 100), wick.map(lambda x: fmt(x, 2)),
         wick.map(lambda x: "Abweisung unten" if x > 0.25 else "Abweisung oben" if x < -0.25 else "neutral")),
    ]

    subs = pd.DataFrame({name: pd.Series(np.asarray(s, dtype=float)) for name, _, s, _, _ in ind})
    groups = {}
    for gname in GROUP_WEIGHTS:
        cols = [name for name, g, *_ in ind if g == gname]
        w = pd.Series({c_: IND_WEIGHTS.get(c_, 1.0) for c_ in cols})
        part = subs[cols]
        groups[gname] = (part.fillna(0) * w).sum(axis=1) / (part.notna() * w).sum(axis=1).replace(0, np.nan)
    lage, wende = groups["Lage"], groups["Wende"]
    dev = (lage - 50) / 50
    turn = ((wende - 50) / 50).fillna(0)
    # Wende bestätigt -> volles Signal; Wende widerspricht -> stark gedämpft ("fallendes Messer")
    gate = (GATE_BASE + GATE_GAIN * turn * np.sign(dev)).clip(0.05, 1.0)
    score = (50 + 50 * dev * gate * STRETCH).clip(0, 100)
    score[: min(len(score), 30)] = np.nan   # Einschwingphase

    last = len(df) - 1
    details = []
    for name, g, _, val, note in ind:
        sv = subs[name].iloc[last]
        if not np.isnan(sv):
            details.append([name, g, round(float(sv)), str(val.iloc[last]), str(note.iloc[last])])
    gl = {g: (None if np.isnan(s.iloc[last]) else round(float(s.iloc[last]))) for g, s in groups.items()}
    return score, details, gl


def pivots(df, n):
    """Echte Hochs/Tiefs – nur im Nachhinein bekannt, daher nur zum Vergleich."""
    w = 2 * n + 1
    hi = df["h"] == df["h"].rolling(w, center=True).max()
    lo = df["l"] == df["l"].rolling(w, center=True).min()
    return hi.fillna(False).values, lo.fillna(False).values


# ----------------------------------------------------------------- Ausgabe
def sig(x, digits=6):
    if x is None or not np.isfinite(x) or x == 0:
        return 0
    return float(f"{x:.{digits}g}")


def build_tf(df):
    out = {}
    score = None
    if len(df) >= SCORE_MIN:
        score, details, groups = analyse(df)
        if score.notna().sum() == 0:
            score = None
        else:
            out["b"], out["k"] = details, groups
    piv_n = PIVOT_N if len(df) >= 60 else 3
    hi, lo = pivots(df, piv_n)
    start = max(0, len(df) - DISPLAY)
    tail = df.iloc[start:]
    out["c"] = [[int(r.t), sig(r.o), sig(r.h), sig(r.l), sig(r.c)] for r in tail.itertuples()]
    out["p"] = [[int(df["t"].iloc[i]), "H" if hi[i] else "T"] for i in range(start, len(df)) if hi[i] or lo[i]]
    if score is not None:
        z = score.iloc[start:]
        out["z"] = [None if np.isnan(x) else round(float(x), 1) for x in z]
        out["stat"] = signal_stats(tail["c"].values, out["z"])
    return out


def signal_stats(closes, scores):
    """Wie entwickelte sich der Kurs 10 Kerzen nach einem Signal (nur sichtbarer Bereich)."""
    res = {}
    for kind in ("buy", "sell"):
        rets = []
        for i in range(1, len(scores) - STAT_HORIZON):
            s, p = scores[i], scores[i - 1]
            if s is None:
                continue
            entry = (s >= BUY_LEVEL and (p is None or p < BUY_LEVEL)) if kind == "buy" else \
                    (s <= SELL_LEVEL and (p is None or p > SELL_LEVEL))
            if entry:
                rets.append(closes[i + STAT_HORIZON] / closes[i] - 1)
        if rets:
            hits = sum(1 for r in rets if (r > 0 if kind == "buy" else r < 0))
            res[kind] = [len(rets), round(100 * float(np.mean(rets)), 2), round(100 * hits / len(rets))]
    return res


def last_score(tfdata):
    z = tfdata.get("z") if tfdata else None
    if not z:
        return None
    return next((x for x in reversed(z) if x is not None), None)


def safe_name(sym):
    return re.sub(r"[^A-Za-z0-9]", "_", sym)


def main():
    started = time.time()
    coins = load_coins()
    os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
    listing, seen = [], set()
    for meta in coins:
        if len(listing) >= TOP_N:
            break
        sym = meta["sym"]
        if not sym or sym in seen:
            continue
        src, frames = load_all(sym)
        if not frames:
            log(f"  {sym}: keine Kerzen gefunden – übersprungen")
            continue
        try:
            tfs = {}
            for key, _ in TIMEFRAMES:
                df = frames.get(key)
                if df is not None and len(df) >= 2:
                    tfs[key] = build_tf(df)
            fname = safe_name(sym)
            with open(os.path.join(OUT_DIR, "data", fname + ".json"), "w", encoding="utf-8") as f:
                json.dump({"s": sym, "src": src, "tf": tfs}, f, separators=(",", ":"))
            seen.add(sym)
            listing.append({
                "s": sym, "n": meta["name"], "r": meta["rank"], "f": fname, "src": src,
                "p": sig(meta["price"] or frames["4h"]["c"].iloc[-1]),
                "ch": None if meta["chg"] is None else round(float(meta["chg"]), 2),
                "sc": {tf: last_score(tfs.get(tf)) for tf in LIST_TFS},
            })
            log(f"  {len(listing):3d}. {sym:<8} 4h-Score {listing[-1]['sc']['4h']}  ({src}, {len(tfs)} Zeitrahmen)")
        except Exception as e:
            log(f"  {sym}: Fehler: {e}")
    if not listing:
        raise SystemExit("Keine Daten – Abbruch.")
    listing.sort(key=lambda c: c["r"])
    payload = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "buy": BUY_LEVEL, "sell": SELL_LEVEL, "h": STAT_HORIZON, "w": GROUP_WEIGHTS,
               "tfs": TIMEFRAMES, "ltfs": LIST_TFS, "iw": IND_WEIGHTS, "info": IND_INFO,
               "gate": [GATE_BASE, GATE_GAIN], "stretch": STRETCH, "coins": listing}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", data))
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"updated": payload["updated"]}, f)
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
header{position:sticky;top:0;z-index:5;background:rgba(11,14,20,.94);backdrop-filter:blur(10px);
padding:calc(10px + env(safe-area-inset-top)) 14px 10px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0 0 2px;font-weight:700}
.sub{color:var(--mut);font-size:12px}
.search{margin-top:10px;width:100%;padding:11px 14px;border-radius:12px;border:1px solid var(--line);
background:var(--card);color:var(--txt);font-size:16px;outline:none}
.search:focus{border-color:var(--acc)}
.chips{display:flex;gap:7px;margin-top:9px;overflow-x:auto;scrollbar-width:none;align-items:center}
.chips::-webkit-scrollbar{display:none}
.chips .lbl{flex:none;font-size:12px;color:var(--mut);margin-right:2px}
.chip{flex:none;padding:7px 12px;border-radius:999px;border:1px solid var(--line);background:var(--card);
color:var(--txt);font-size:13px;font-weight:600;font-family:inherit}
.chip.on{background:var(--txt);color:var(--bg);border-color:var(--txt)}
.chip.b.on{background:var(--buy);border-color:var(--buy);color:#04130a}
.chip.s.on{background:var(--sell);border-color:var(--sell);color:#fff}
.chip:disabled{opacity:.35}
.stats{display:flex;gap:8px;padding:12px 14px 4px}
.stat{flex:1;background:var(--card);border-radius:12px;padding:8px 10px}
.stat b{display:block;font-size:18px}.stat span{font-size:11px;color:var(--mut)}
ul{list-style:none;margin:0;padding:6px 10px 30px}
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
.dh{position:sticky;top:0;z-index:3;background:rgba(11,14,20,.96);backdrop-filter:blur(10px);
padding:calc(10px + env(safe-area-inset-top)) 12px 8px;border-bottom:1px solid var(--line)}
.dh .top{display:flex;align-items:center;gap:10px}
.back{border:0;background:var(--card);color:var(--txt);width:38px;height:38px;border-radius:12px;font-size:22px}
.dh .t{flex:1;min-width:0}.dh .t b{font-size:17px}.dh .t div{font-size:12px;color:var(--mut)}
.box{margin:10px;background:var(--card);border-radius:16px;padding:10px 6px 6px;position:relative}
.box h3{font-size:13px;margin:2px 8px 6px;color:var(--mut);font-weight:600;display:flex;justify-content:space-between;gap:8px}
.legend{font-size:11.5px;color:var(--mut);margin:0 8px 4px;min-height:16px}
#pc{height:320px}#zc{height:170px}
.msg{color:var(--mut);font-size:13px;padding:14px 10px;text-align:center}
.sec{margin:10px;background:var(--card);border-radius:16px;padding:12px}
.sec h4{margin:0 0 8px;font-size:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
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
.hrow{display:flex;align-items:center;gap:10px}.hrow>div{flex:1;min-width:0}
.rbtn{flex:none;border:1px solid var(--line);background:var(--card);color:var(--txt);height:38px;border-radius:12px;
padding:0 12px;font-size:13px;font-weight:600;font-family:inherit;display:flex;align-items:center;gap:6px}
.rbtn.spin span{display:inline-block;animation:sp 1s linear infinite}@keyframes sp{to{transform:rotate(360deg)}}
.live{font-size:11px;color:var(--acc)}
.pcw{position:relative}
#mlayer{position:absolute;inset:0;pointer-events:none;z-index:2;overflow:hidden}
#mlayer .r{position:absolute;border:1px dashed}
#mlayer .d{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;background:var(--acc)}
.mres{margin:6px 8px 2px;font-size:13px;min-height:0}
.mres b{font-size:15px}
.hint{color:var(--acc)}
.exp p{font-size:12.5px;color:var(--mut);margin:6px 0}.exp b{color:var(--txt)}
.exp li{font-size:12px;color:var(--mut);margin:4px 0}.exp ul{padding:0 0 0 16px;list-style:disc}
.note{font-size:11px;color:var(--mut);margin:8px 12px 0;line-height:1.5}
</style>
</head>
<body>
<header>
  <div class="hrow"><div><h1>Krypto Score Dashboard</h1>
  <div class="sub" id="upd"></div><div class="live" id="live"></div></div>
  <button class="rbtn" id="ref"><span>⟳</span> Aktualisieren</button></div>
  <input class="search" id="q" type="search" placeholder="Coin suchen (z. B. BTC, Solana)…" autocomplete="off">
  <div class="chips" id="sort"><span class="lbl">Sortieren:</span>
    <button class="chip on" data-o="mc">Marktkap.</button>
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
<footer> Score hoch = Kurs weit unten mit Wende-Zeichen (Kaufzone),
Score tief = Kurs weit oben mit Schwäche-Zeichen (Verkaufszone).<br>Keine Anlageberatung.</footer>

<div id="detail">
  <div class="dh">
    <div class="top">
      <button class="back" id="back" aria-label="Zurück">‹</button>
      <div class="t"><b id="dt"></b><div id="ds"></div></div>
      <button class="rbtn" id="ref2" aria-label="Aktualisieren"><span>⟳</span></button>
      <div class="sc" id="dsc"></div>
    </div>
    <div class="chips" id="tfc"></div>
  </div>
  <div class="box"><h3><span>Kurs</span>
    <span><button class="chip" id="mbtn" style="padding:4px 10px;font-size:12px">📏 Messen</button>
    <button class="chip" id="pivbtn" style="padding:4px 10px;font-size:12px">Hochs/Tiefs</button></span></h3>
    <div class="legend" id="pl"></div><div class="mres" id="mres"></div>
    <div class="pcw"><div id="pc"></div><div id="mlayer"></div></div></div>
  <div class="box" id="zbox"><h3><span>Score 0–100</span><span id="zl"></span></h3><div id="zc"></div></div>
  <div class="sec" id="nobox" style="display:none"><div class="msg">Für diesen Zeitrahmen gibt es zu wenige Kerzen
    für einen Score – angezeigt werden nur die Kerzen.</div></div>
  <div class="sec" id="statbox"><h4>Bilanz der Signale im Chart</h4><div class="grid2" id="stat"></div></div>
  <div class="sec" id="catbox"><h4>Bestandteile</h4><div class="grid2" id="cats"></div></div>
  <div class="sec" id="pcbox"><div class="grid2">
    <div><h4 class="up">Spricht für Kaufen</h4><div id="pro"></div></div>
    <div><h4 class="dn">Spricht für Verkaufen</h4><div id="con"></div></div>
  </div></div>
  <div class="sec" id="allbox"><details><summary>Alle Indikatoren</summary><table id="all"></table></details></div>
  <div class="sec exp"><details><summary>Wie entsteht der Score?</summary><div class="expc"></div></details></div>
  <p class="note">▲ Kauf / ▼ Verkauf = erste Kerze, mit der der Score in die Zone eintritt. Der Score nutzt nur Daten
  bis zur jeweiligen Kerze; nur die letzte, noch laufende Kerze kann sich ändern.
  „Echte Hochs/Tiefs“ (graue Punkte) stehen erst im Nachhinein fest und dienen nur zum Vergleich.
  Datenquelle: <span id="src"></span>. Keine Anlageberatung.</p>
</div>

<script>window.DATA = __DATA__;</script>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
 onerror="var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js';document.head.appendChild(s)"></script>
<script>
(function(){
const D = window.DATA, BUY = D.buy, SELL = D.sell;
const TFL = Object.fromEntries(D.tfs);
const $ = id => document.getElementById(id);
const store = (k, v) => { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch(e) { return null; } };
let sortBy = store('sort') || 'mc';
let filter = 'all', query = '', ltf = store('ltf') || '4h', ctf = store('ctf') || '4h', showPiv = store('piv') === '1';
let cur = null, charts = [], reqId = 0;
const cache = {};

const color = s => s == null ? '#4b5563' : s >= BUY ? '#22c55e' : s >= 60 ? '#84cc16' : s > 40 ? '#eab308' : s > SELL ? '#f97316' : '#ef4444';
const label = s => s == null ? 'kein Score' : s >= BUY ? 'Starker Kauf' : s >= 60 ? 'Kauf' : s > 40 ? 'Neutral' : s > SELL ? 'Verkauf' : 'Starker Verkauf';
function price(p){
  if (p == null) return '–';
  const d = p >= 1000 ? 0 : p >= 1 ? 2 : p >= 0.01 ? 4 : 8;
  return '$' + p.toLocaleString('de-DE', {minimumFractionDigits: d, maximumFractionDigits: d});
}
const pct = x => x == null ? '' : (x >= 0 ? '+' : '') + x.toFixed(2).replace('.', ',') + ' %';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

$('upd').textContent = 'Stand: ' + new Date(D.updated).toLocaleString('de-DE', {dateStyle:'medium', timeStyle:'short'}) +
  ' · ' + D.coins.length + ' Coins · stündliches Update';

// --- Liste
$('ltf').insertAdjacentHTML('beforeend', D.ltfs.map(t => `<button class="chip" data-t="${t}">${TFL[t]}</button>`).join(''));
function setLtf(t){ ltf = t; store('ltf', t); document.querySelectorAll('#ltf .chip').forEach(b => b.classList.toggle('on', b.dataset.t === t)); render(); }
$('ltf').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setLtf(b.dataset.t); });

function match(c){
  if (query && !(c.s.toLowerCase().includes(query) || c.n.toLowerCase().includes(query))) return false;
  const s = c.sc[ltf];
  if (filter !== 'all' && s == null) return false;
  switch (filter){
    case 'buy': return s >= BUY;
    case 'pos': return s >= 60 && s < BUY;
    case 'neu': return s > 40 && s < 60;
    case 'neg': return s > SELL && s <= 40;
    case 'sell': return s <= SELL;
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
  const sv = c => c.sc[ltf] == null ? null : c.sc[ltf];
  if (sortBy === 'mc') rows.sort((a, b) => a.r - b.r);
  else rows.sort((a, b) => { const x = sv(a), y = sv(b);
    if (x == null) return 1; if (y == null) return -1; return sortBy === 'hi' ? y - x : x - y; });
  $('list').innerHTML = rows.length ? rows.map(c => {
    const s = c.sc[ltf], col = color(s);
    return `<li class="row" data-s="${esc(c.s)}">
      <div class="rk">#${c.r}</div>
      <div class="nm"><b>${esc(c.s)}</b><small>${esc(c.n)}</small>
        <div class="px">${price(c.p)} <span class="${c.ch >= 0 ? 'up' : 'dn'}">${pct(c.ch)}</span> · ${label(s)}</div>
        <div class="bar"><i style="width:${s || 0}%;background:${col}"></i></div></div>
      <div class="sc" style="color:${col};background:${col}22">${s == null ? '–' : Math.round(s)}</div></li>`;
  }).join('') : '<div class="empty">Keine Coins in diesem Filter.</div>';
}
function setSort(o){ sortBy = o; store('sort', o); document.querySelectorAll('#sort .chip').forEach(b => b.classList.toggle('on', b.dataset.o === o)); render(); }
$('sort').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setSort(b.dataset.o); });
$('q').addEventListener('input', e => { query = e.target.value.trim().toLowerCase(); render(); });
$('chips').addEventListener('click', e => {
  const b = e.target.closest('.chip'); if (!b) return;
  document.querySelectorAll('#chips .chip').forEach(x => x.classList.toggle('on', x === b));
  filter = b.dataset.f; render();
});
$('list').addEventListener('click', e => {
  const li = e.target.closest('li.row');
  if (li) { history.pushState({c: li.dataset.s}, '', '#' + encodeURIComponent(li.dataset.s)); openCoin(li.dataset.s); }
});
$('back').addEventListener('click', () => location.hash ? history.back() : closeCoin());
window.addEventListener('popstate', () => { const s = decodeURIComponent(location.hash.slice(1)); s ? openCoin(s) : closeCoin(); });
document.querySelectorAll('#sort .chip').forEach(b => b.classList.toggle('on', b.dataset.o === sortBy));
setLtf(D.ltfs.includes(ltf) ? ltf : '4h');

// --- Erklärung
(function(){
  const lage = Object.keys(D.info).filter((k, i) => i < 9), wende = Object.keys(D.info).filter((k, i) => i >= 9);
  const li = k => `<li><b>${esc(k)}</b> (Gewicht ${String(D.iw[k] || 1).replace('.', ',')}): ${esc(D.info[k])}</li>`;
  const html = `
  <p>Jeder Indikator wird für jede Kerze in einen <b>Teil-Score von 0 bis 100</b> umgerechnet
  (100 = spricht für Kaufen, 0 = für Verkaufen). Die Teil-Scores bilden zwei Gruppen:</p>
  <p><b>1. Lage (${Math.round(D.w.Lage * 100)} %)</b> – Wie weit unten oder oben steht der Kurs gerade?</p><ul>${lage.map(li).join('')}</ul>
  <p><b>2. Wende (${Math.round(D.w.Wende * 100)} %)</b> – Gibt es erste Zeichen, dass der Kurs dreht?</p><ul>${wende.map(li).join('')}</ul>
  <p><b>So werden sie verbunden:</b> Die Lage bestimmt die Richtung und Stärke. Die Wende wirkt als <b>Bestätigung</b>:
  Liegt der Kurs weit unten und dreht schon nach oben, zählt das Signal voll. Fällt er noch weiter („fallendes Messer“),
  wird das Signal auf bis zu ${Math.round((D.gate[0] - D.gate[1]) * 100) < 5 ? 5 : Math.round((D.gate[0] - D.gate[1]) * 100)} % gedämpft.
  Danach wird das Ergebnis um 50 gespreizt (Faktor ${String(D.stretch).replace('.', ',')}).</p>
  <p><b>Zonen:</b> ≥ ${BUY} = Kaufzone (▲), ≤ ${SELL} = Verkaufszone (▼), um 50 = neutral.</p>
  <p><b>Ehrlichkeit:</b> Für jede Kerze werden nur Daten bis zu dieser Kerze benutzt. Alte Signale ändern sich nicht nachträglich.
  Die grauen „Hochs/Tiefs“ stehen erst im Nachhinein fest (±10 Kerzen) und zeigen nur, wie gut die Signale getroffen hätten.
  Die „Bilanz“ zeigt, wie sich der Kurs ${D.h} Kerzen nach jedem Signal im sichtbaren Chart entwickelt hat.</p>
  <p>Kein Indikator kann Tiefs und Hochs sicher vorhersagen. Keine Anlageberatung.</p>`;
  document.querySelectorAll('.expc').forEach(el => el.innerHTML = html);
})();

// --- Aktualisieren (neuer Server-Stand + Live-Preise von Binance direkt im Browser)
const HOSTS = ['https://data-api.binance.vision', 'https://api.binance.com'];
async function bget(path){
  let err;
  for (const h of HOSTS){
    try { const r = await fetch(h + path, {cache: 'no-store'}); if (r.ok) return await r.json(); err = r.status; if (r.status === 400) break; }
    catch (e) { err = e; }
  }
  throw new Error(err);
}
let lastLive = null;
async function refresh(){
  const btns = [$('ref'), $('ref2')]; btns.forEach(b => b.classList.add('spin'));
  try {
    try {
      const m = await (await fetch('meta.json?t=' + Date.now(), {cache: 'no-store'})).json();
      if (m.updated && m.updated !== D.updated) { location.reload(); return; }
    } catch (e) {}
    const all = await bget('/api/v3/ticker/24hr?type=MINI');
    const map = {}; all.forEach(t => map[t.symbol] = t);
    let n = 0;
    D.coins.forEach(c => { const t = map[c.s + 'USDT']; if (!t) return;
      const last = +t.lastPrice, open = +t.openPrice; c.p = last; if (open) c.ch = (last / open - 1) * 100; n++; });
    lastLive = map;
    $('live').textContent = 'Live-Preise: ' + new Date().toLocaleTimeString('de-DE', {hour: '2-digit', minute: '2-digit', second: '2-digit'}) + ' (' + n + ' Coins)';
    render();
    if (cur) await liveChart();
  } catch (e) {
    $('live').textContent = 'Live-Preise gerade nicht abrufbar.';
  } finally { btns.forEach(b => b.classList.remove('spin')); }
}
$('ref').addEventListener('click', refresh);
$('ref2').addEventListener('click', refresh);

const NATIVE = ['1h', '4h', '1d', '1w', '1M'];
async function liveChart(){
  const meta = cur.meta, T = cur.data.tf[ctf]; if (!T) return;
  const sym = meta.s + 'USDT';
  try {
    if (NATIVE.includes(ctf)){
      const k = await bget(`/api/v3/klines?symbol=${sym}&interval=${ctf}&limit=6`);
      k.forEach(r => {
        const t = Math.floor(r[0] / 1000), row = [t, +r[1], +r[2], +r[3], +r[4]];
        const lastT = T.c[T.c.length - 1][0];
        if (t === lastT) T.c[T.c.length - 1] = row;
        else if (t > lastT) { T.c.push(row); if (T.z) T.z.push(null); }
        else { const i = T.c.findIndex(x => x[0] === t); if (i >= 0) T.c[i] = row; }
      });
    } else {
      const t = lastLive && lastLive[sym] ? lastLive[sym] : null;
      const p = t ? +t.lastPrice : +(await bget(`/api/v3/ticker/price?symbol=${sym}`)).price;
      const L = T.c[T.c.length - 1]; L[4] = p; L[2] = Math.max(L[2], p); L[3] = Math.min(L[3], p);
    }
    const p = T.c[T.c.length - 1][4]; meta.p = p;
    $('ds').innerHTML = price(meta.p) + ' <span class="' + (meta.ch >= 0 ? 'up' : 'dn') + '">' + pct(meta.ch) + '</span> · Rang #' + meta.r + ' · <span class="live">live</span>';
    show(true);
  } catch (e) { $('pl').textContent = 'Live-Kerzen für ' + meta.s + ' nicht bei Binance verfügbar.'; }
}

// --- Detail
function destroy(){ charts.forEach(c => c.remove()); charts = []; }
function closeCoin(){ destroy(); cur = null; $('detail').classList.remove('open'); document.body.style.overflow = ''; }

$('tfc').innerHTML = D.tfs.map(([k, l]) => `<button class="chip" data-t="${k}">${l}</button>`).join('');
$('tfc').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b && !b.disabled) { ctf = b.dataset.t; store('ctf', ctf); show(); } });
$('pivbtn').addEventListener('click', () => { showPiv = !showPiv; store('piv', showPiv ? '1' : '0'); show(true); });

// --- Messwerkzeug
const measure = {on: false, a: null, b: null};
const SEC = {'1h': 3600, '4h': 14400, '1d': 86400, '5d': 432000, '1w': 604800, '2w': 1209600, '1M': 2629800, '6M': 15778800, '1y': 31557600};
let mctx = null;
$('mbtn').addEventListener('click', () => {
  measure.on = !measure.on; measure.a = measure.b = null;
  $('mbtn').classList.toggle('on', measure.on); drawMeasure();
});
function dur(sec){
  const a = Math.abs(sec), d = Math.floor(a / 86400), h = Math.round((a % 86400) / 3600);
  if (d >= 365) return '≈ ' + (a / 31557600).toFixed(1).replace('.', ',') + ' Jahre';
  if (d >= 60) return '≈ ' + (a / 2629800).toFixed(1).replace('.', ',') + ' Monate';
  if (d >= 1) return d + ' T' + (h ? ' ' + h + ' Std' : '');
  return h + ' Std';
}
function drawMeasure(){
  const layer = $('mlayer'); layer.innerHTML = '';
  const res = $('mres');
  if (!measure.on) { res.innerHTML = ''; return; }
  if (!measure.a) { res.innerHTML = '<span class="hint">📏 Tippe in den Chart auf den Startpunkt.</span>'; return; }
  if (!mctx) return;
  const {chart, series} = mctx, ts = chart.timeScale();
  const pos = q => ({x: ts.logicalToCoordinate(q.l), y: series.priceToCoordinate(q.p)});
  const A = pos(measure.a);
  if (!measure.b) {
    if (A.x != null && A.y != null) layer.innerHTML = `<div class="d" style="left:${A.x}px;top:${A.y}px"></div>`;
    res.innerHTML = '<span class="hint">📏 Jetzt auf den Endpunkt tippen.</span>'; return;
  }
  const B = pos(measure.b), up = measure.b.p >= measure.a.p, col = up ? '#22c55e' : '#ef4444';
  if (A.x != null && B.x != null && A.y != null && B.y != null)
    layer.innerHTML = `<div class="r" style="left:${Math.min(A.x, B.x)}px;top:${Math.min(A.y, B.y)}px;width:${Math.abs(B.x - A.x)}px;height:${Math.abs(B.y - A.y)}px;border-color:${col};background:${col}26"></div>
      <div class="d" style="left:${A.x}px;top:${A.y}px"></div><div class="d" style="left:${B.x}px;top:${B.y}px;background:${col}"></div>`;
  const dp = measure.b.p - measure.a.p, pc_ = dp / measure.a.p * 100, bars = measure.b.l - measure.a.l;
  const fmtP = v => (v >= 0 ? '+' : '−') + price(Math.abs(v)).replace('$', '$');
  res.innerHTML = `<b style="color:${col}">${pct(pc_)}</b> &nbsp;(${fmtP(dp)}) &nbsp;·&nbsp; <b>${Math.abs(bars)}</b> Kerzen · ${dur(bars * SEC[ctf])}
    <span style="color:var(--mut)"> · erneut tippen = neue Messung</span>`;
}

async function openCoin(sym){
  const meta = D.coins.find(x => x.s === sym); if (!meta) return closeCoin();
  destroy();
  $('detail').classList.add('open'); $('detail').scrollTop = 0; document.body.style.overflow = 'hidden';
  $('dt').textContent = meta.s + ' · ' + meta.n;
  $('ds').innerHTML = price(meta.p) + ' <span class="' + (meta.ch >= 0 ? 'up' : 'dn') + '">' + pct(meta.ch) + '</span> · Rang #' + meta.r;
  $('src').textContent = meta.src;
  $('pl').textContent = 'Lade Daten…';
  const id = ++reqId;
  try {
    if (!cache[meta.f]) {
      const r = await fetch('data/' + meta.f + '.json?v=' + encodeURIComponent(D.updated));
      if (!r.ok) throw new Error(r.status);
      cache[meta.f] = await r.json();
    }
  } catch (e) { if (id === reqId) $('pl').textContent = 'Daten konnten nicht geladen werden.'; return; }
  if (id !== reqId) return;
  cur = {meta, data: cache[meta.f]};
  show();
}

function waitLib(cb, n){ if (window.LightweightCharts) cb(); else if ((n || 0) < 100) setTimeout(() => waitLib(cb, (n || 0) + 1), 100); }

function show(keep){
  if (!cur) return;
  if (!keep) { measure.a = measure.b = null; }
  const tfs = cur.data.tf;
  if (!tfs[ctf]) ctf = tfs['4h'] ? '4h' : Object.keys(tfs)[0];
  document.querySelectorAll('#tfc .chip').forEach(b => { b.disabled = !tfs[b.dataset.t]; b.classList.toggle('on', b.dataset.t === ctf); });
  $('pivbtn').classList.toggle('on', showPiv);
  const T = tfs[ctf], hasZ = !!T.z;
  const s = hasZ ? T.z.slice().reverse().find(x => x != null) : null;
  const col = color(s);
  $('dsc').textContent = s == null ? '–' : Math.round(s); $('dsc').style.color = col; $('dsc').style.background = col + '22';
  $('zl').textContent = label(s) + ' · ' + TFL[ctf];
  ['zbox', 'catbox', 'pcbox', 'allbox', 'statbox'].forEach(id => $(id).style.display = hasZ ? '' : 'none');
  $('nobox').style.display = hasZ ? 'none' : '';
  if (hasZ) details(T);
  destroy();
  waitLib(() => draw(T));
}

function details(T){
  const st = T.stat || {}, H = D.h;
  const box = (k, name, good) => {
    const v = st[k];
    if (!v) return `<div class="cat">${name}<b>–</b>keine Signale im Chart</div>`;
    return `<div class="cat">${name} (${v[0]}×)<b style="color:${v[2] >= 50 ? 'var(--buy)' : 'var(--sell)'}">${v[2]} % Treffer</b>
      Ø ${pct(v[1])} nach ${H} Kerzen</div>`;
  };
  $('stat').innerHTML = box('buy', '▲ Kauf-Signale') + box('sell', '▼ Verkaufs-Signale');
  $('cats').innerHTML = Object.keys(D.w).map(k => {
    const v = T.k[k]; if (v == null) return '';
    const txt = k === 'Lage' ? (v >= 60 ? 'Kurs weit unten' : v <= 40 ? 'Kurs weit oben' : 'Mitte') :
                              (v >= 60 ? 'dreht nach oben' : v <= 40 ? 'dreht nach unten' : 'keine Wende');
    return `<div class="cat">${k} (${Math.round(D.w[k] * 100)} %)<b style="color:${color(v)}">${v}</b>${txt}
      <div class="bar"><i style="width:${v}%;background:${color(v)}"></i></div></div>`;
  }).join('');
  const sorted = [...T.b].sort((a, b) => b[2] - a[2]);
  const item = x => `<div class="it"><b><span style="color:var(--txt)">${esc(x[0])}</span><span style="color:${color(x[2])}">${x[2]}</span></b><span>${esc(x[4])}</span></div>`;
  const pro = sorted.filter(x => x[2] >= 60).slice(0, 4), con = sorted.filter(x => x[2] <= 40).reverse().slice(0, 4);
  $('pro').innerHTML = pro.length ? pro.map(item).join('') : '<div class="it"><span>Keine deutlichen Kaufsignale</span></div>';
  $('con').innerHTML = con.length ? con.map(item).join('') : '<div class="it"><span>Keine deutlichen Verkaufssignale</span></div>';
  $('all').innerHTML = Object.keys(D.w).map(g => `<tr><td colspan="3" style="color:var(--mut);font-weight:600">${g}</td></tr>` +
    T.b.filter(x => x[1] === g).map(x =>
    `<tr><td>${esc(x[0])}<br><span style="color:var(--mut);font-size:11px">${esc(x[4])}</span></td>
     <td class="v">${esc(x[3])}</td>
     <td class="v"><span class="mini"><i style="width:${x[2]}%;background:${color(x[2])}"></i></span> <b style="color:${color(x[2])}">${x[2]}</b></td></tr>`).join('')).join('');
}

function fmtDate(t, long){
  const d = new Date(t * 1000);
  const big = ['1w', '2w', '1M', '6M', '1y', '5d', '1d'].includes(ctf);
  if (big) return d.toLocaleDateString('de-DE', {day: '2-digit', month: '2-digit', year: long ? 'numeric' : '2-digit'});
  return d.toLocaleString('de-DE', {day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'});
}

function draw(T){
  const L = window.LightweightCharts;
  const timeScale = {borderColor: '#242b38', timeVisible: !['1d', '5d', '1w', '2w', '1M', '6M', '1y'].includes(ctf),
    secondsVisible: false, rightOffset: 4, minBarSpacing: 2,
    tickMarkFormatter: (t, type) => {
      const d = new Date(t * 1000);
      if (type === 0) return String(d.getFullYear());
      if (type === 1) return d.toLocaleDateString('de-DE', {month: 'short'});
      if (type === 2) return d.getDate() + '.' + (d.getMonth() + 1) + '.';
      return d.toLocaleTimeString('de-DE', {hour: '2-digit', minute: '2-digit'});
    }};
  const base = {
    autoSize: true,
    layout: {background: {type: 'solid', color: '#131821'}, textColor: '#8b93a5', fontSize: 11},
    grid: {vertLines: {color: '#1c2230'}, horzLines: {color: '#1c2230'}},
    rightPriceScale: {borderColor: '#242b38', minimumWidth: 66},
    timeScale, crosshair: {mode: L.CrosshairMode.Normal},
    localization: {locale: 'de-DE', timeFormatter: t => fmtDate(t, true)},
  };
  const pc = L.createChart($('pc'), base);
  charts = [pc];
  const n = T.c.length, last = T.c[n - 1][4];
  const prec = last >= 1000 ? 2 : last >= 1 ? 3 : last >= 0.01 ? 5 : 8;
  const cs = pc.addCandlestickSeries({upColor: '#22c55e', downColor: '#ef4444', borderVisible: false,
    wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    priceFormat: {type: 'price', precision: prec, minMove: Math.pow(10, -prec)}});
  cs.setData(T.c.map(k => ({time: k[0], open: k[1], high: k[2], low: k[3], close: k[4]})));

  const markers = [];
  if (T.z) for (let i = 1; i < T.z.length; i++){
    const s = T.z[i], p = T.z[i - 1]; if (s == null) continue;
    if (s >= BUY && (p == null || p < BUY)) markers.push({time: T.c[i][0], position: 'belowBar', color: '#22c55e', shape: 'arrowUp', text: 'Kauf'});
    if (s <= SELL && (p == null || p > SELL)) markers.push({time: T.c[i][0], position: 'aboveBar', color: '#ef4444', shape: 'arrowDown', text: 'Verkauf'});
  }
  if (showPiv) T.p.forEach(([t, k]) => markers.push({time: t, position: k === 'H' ? 'aboveBar' : 'belowBar', color: '#9ca3af', shape: 'circle', size: 0.6}));
  markers.sort((a, b) => a.time - b.time);
  cs.setMarkers(markers);

  let zc = null, zs = null;
  if (T.z){
    zc = L.createChart($('zc'), base);
    charts.push(zc);
    zs = zc.addBaselineSeries({baseValue: {type: 'price', price: 50}, lineWidth: 2,
      topLineColor: '#22c55e', topFillColor1: 'rgba(34,197,94,.28)', topFillColor2: 'rgba(34,197,94,.02)',
      bottomLineColor: '#ef4444', bottomFillColor1: 'rgba(239,68,68,.02)', bottomFillColor2: 'rgba(239,68,68,.28)',
      priceFormat: {type: 'price', precision: 0, minMove: 1},
      autoscaleInfoProvider: () => ({priceRange: {minValue: 0, maxValue: 100}})});
    zs.setData(T.c.map((k, i) => T.z[i] == null ? {time: k[0]} : {time: k[0], value: T.z[i]}));
    zs.createPriceLine({price: BUY, color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Kauf'});
    zs.createPriceLine({price: SELL, color: '#ef4444', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Verkauf'});
    zs.createPriceLine({price: 50, color: '#4b5563', lineWidth: 1, lineStyle: 3, axisLabelVisible: false});
  }

  const vis = Math.min(n, 120);
  const range = {from: n - vis - 0.5, to: n + 3};
  charts.forEach(c => c.timeScale().setVisibleLogicalRange(range));

  let lock = false;
  if (zc){
    const link = (a, b) => a.timeScale().subscribeVisibleLogicalRangeChange(r => {
      if (!r || lock) return; lock = true; b.timeScale().setVisibleLogicalRange(r); lock = false;
    });
    link(pc, zc); link(zc, pc);
  }
  mctx = {chart: pc, series: cs};
  pc.subscribeClick(p => {
    if (!measure.on || !p.point) return;
    const lg = pc.timeScale().coordinateToLogical(p.point.x), pr = cs.coordinateToPrice(p.point.y);
    if (lg == null || pr == null) return;
    const pt = {l: Math.round(lg), p: pr};
    if (!measure.a || measure.b) { measure.a = pt; measure.b = null; } else measure.b = pt;
    drawMeasure();
  });
  pc.timeScale().subscribeVisibleLogicalRangeChange(() => requestAnimationFrame(drawMeasure));
  requestAnimationFrame(drawMeasure);
  const byTime = {}; T.c.forEach((k, i) => byTime[k[0]] = i);
  const legend = i => {
    const k = T.c[i], s = T.z ? T.z[i] : null;
    $('pl').innerHTML = `<b style="color:var(--txt)">${fmtDate(k[0], true)}</b> · O ${k[1]} H ${k[2]} L ${k[3]} C ${k[4]}` +
      (s != null ? ` · <b style="color:${color(s)}">Score ${Math.round(s)}</b>` : '');
  };
  legend(n - 1);
  const onMove = (other, otherSeries, val) => p => {
    if (lock) return;
    if (!p.time) { legend(n - 1); if (other) { lock = true; other.clearCrosshairPosition(); lock = false; } return; }
    const i = byTime[p.time]; if (i == null) return; legend(i);
    if (other){ const v = val(i); lock = true; if (v != null) other.setCrosshairPosition(v, p.time, otherSeries); lock = false; }
  };
  pc.subscribeCrosshairMove(onMove(zc, zs, i => T.z && T.z[i]));
  if (zc) zc.subscribeCrosshairMove(onMove(pc, cs, i => T.c[i][4]));
}

const h0 = decodeURIComponent(location.hash.slice(1));
if (h0) openCoin(h0);
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
