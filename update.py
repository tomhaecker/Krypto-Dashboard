#!/usr/bin/env python3
"""
Krypto-Dashboard – wird automatisch von GitHub Actions ausgeführt.

1. Holt die Top-Coins nach Marktkapitalisierung (CoinGecko, Ersatz: CoinPaprika)
   und filtert Stablecoins sowie Wrapped/Staked-Kopien heraus.
2. Lädt pro Coin 4-Stunden-Kerzen (Binance, Ersatz: OKX, KuCoin – alles ohne API-Key).
3. Berechnet für jede der letzten 120 Kerzen einen Score von 0–100 aus 18 Indikatoren.
4. Schreibt eine eigenständige Seite nach site/index.html.

Keine Anlageberatung – die Scores sind rein technische Kennzahlen.
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------- Einstellungen
TOP_N = 100            # Anzahl Coins im Dashboard
HISTORY = 120          # Anzahl 4h-Kerzen in Chart und Score-Kurve
FETCH_CANDLES = 500    # Geladene Kerzen (Rest dient als Vorlauf für EMA/SMA 200)
MIN_CANDLES = HISTORY + 60
CANDIDATES = 250       # So viele Coins werden geprüft, um 100 brauchbare zu finden
BUY_LEVEL = 80
SELL_LEVEL = 20
STRETCH = 2.0          # Spreizt den Score um 50, da sich Indikatoren oft gegenseitig aufheben
OUT_FILE = "site/index.html"

CATEGORY_WEIGHTS = {"Trend": 0.35, "Momentum": 0.30, "Volumen": 0.20, "Volatilität": 0.15}

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
    "CBETH", "SAVAX", "WBNB", "WTRX", "STKAITER", "TBTC", "JSOL", "SWETH", "PUFETH", "LSETH",
    "FBTC", "UNIBTC", "PUMPBTC", "EBTC", "SFRXETH", "FRXETH", "WHYPE", "KHYPE", "STHYPE",
    "BBSOL", "HSOL", "STS", "WS", "XSOLVBTC", "ENABTC", "WEETHS", "RSWETH", "ETHX", "TETH",
}
NAME_SKIP = ("wrapped", "staked", "bridged", "restaked", "liquid staking", "tokenized", "bitcoin bep2")

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "krypto-dashboard/1.0 (+github-actions)"
SESSION.headers["Accept"] = "application/json"


class Blocked(Exception):
    """Börse ist von diesem Server aus gesperrt (z. B. HTTP 451 in den USA)."""


class NotFound(Exception):
    """Handelspaar existiert an dieser Börse nicht."""


def log(*a):
    print(*a, flush=True)


def get_json(url, params=None, tries=4, timeout=20):
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After") or 0) or 15 * (i + 1)
                log(f"  Rate-Limit bei {url} – warte {wait}s")
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
        except Exception as e:  # Netzwerkfehler -> nochmal versuchen
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
        log("  Stablecoin-Kategorie nicht abrufbar, nutze nur feste Liste:", e)
    coins = []
    for c in rows:
        coins.append({
            "id": c.get("id"), "sym": (c.get("symbol") or "").upper(), "name": c.get("name") or "",
            "rank": c.get("market_cap_rank") or 9999, "price": c.get("current_price"),
            "mcap": c.get("market_cap"), "chg": c.get("price_change_percentage_24h"),
            "stable": c.get("id") in stable_ids,
        })
    return coins


def coins_from_coinpaprika():
    rows = get_json("https://api.coinpaprika.com/v1/tickers", {"quotes": "USD"}, timeout=40)
    rows = sorted([r for r in rows if r.get("rank")], key=lambda r: r["rank"])[:CANDIDATES]
    coins = []
    for c in rows:
        q = (c.get("quotes") or {}).get("USD", {})
        coins.append({
            "id": c.get("id"), "sym": (c.get("symbol") or "").upper(), "name": c.get("name") or "",
            "rank": c.get("rank"), "price": q.get("price"), "mcap": q.get("market_cap"),
            "chg": q.get("percent_change_24h"), "stable": False,
        })
    return coins


def is_excluded(c):
    sym, name, price = c["sym"], c["name"].lower(), c["price"] or 0
    if c["stable"] or sym in STABLE_SYMBOLS or sym in WRAPPED_SYMBOLS:
        return True
    if any(k in name for k in NAME_SKIP) or "stablecoin" in name:
        return True
    if ("USD" in sym or "usd" in name) and 0.95 < price < 1.05:
        return True
    return False


def load_coins():
    for fn in (coins_from_coingecko, coins_from_coinpaprika):
        try:
            coins = fn()
            if coins:
                log(f"Coin-Liste über {fn.__name__}: {len(coins)} Kandidaten")
                return [c for c in coins if not is_excluded(c)]
        except Exception as e:
            log(f"{fn.__name__} fehlgeschlagen: {e}")
    raise SystemExit("Keine Coin-Liste abrufbar – Abbruch.")


# ----------------------------------------------------------------- Kerzen
def _finish(rows):
    """Sortiert, entfernt die noch laufende Kerze und baut ein DataFrame."""
    now = time.time()
    rows = sorted({r[0]: r for r in rows if r[0] + 4 * 3600 <= now}.values())
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    return df.astype({"t": "int64", "o": float, "h": float, "l": float, "c": float, "v": float})


def candles_binance(base, sym):
    data = get_json(f"{base}/api/v3/klines",
                    {"symbol": f"{sym}USDT", "interval": "4h", "limit": FETCH_CANDLES})
    return _finish([(int(k[0]) // 1000, k[1], k[2], k[3], k[4], k[5]) for k in data])


def candles_okx(sym):
    data = get_json("https://www.okx.com/api/v5/market/candles",
                    {"instId": f"{sym}-USDT", "bar": "4Hutc", "limit": 300})
    if str(data.get("code")) != "0" or not data.get("data"):
        raise NotFound(data.get("msg") or "leer")
    return _finish([(int(k[0]) // 1000, k[1], k[2], k[3], k[4], k[5]) for k in data["data"]])


def candles_kucoin(sym):
    end = int(time.time())
    data = get_json("https://api.kucoin.com/api/v1/market/candles",
                    {"type": "4hour", "symbol": f"{sym}-USDT",
                     "startAt": end - FETCH_CANDLES * 4 * 3600, "endAt": end})
    if str(data.get("code")) != "200000" or not data.get("data"):
        raise NotFound(data.get("msg") or "leer")
    # KuCoin-Reihenfolge: Zeit, Open, Close, High, Low, Volumen
    return _finish([(int(k[0]), k[1], k[3], k[4], k[2], k[5]) for k in data["data"]])


SOURCES = [
    ("Binance", lambda s: candles_binance("https://data-api.binance.vision", s)),
    ("Binance", lambda s: candles_binance("https://api.binance.com", s)),
    ("OKX", candles_okx),
    ("KuCoin", candles_kucoin),
]
DEAD_SOURCES = set()


def load_candles(sym):
    for i, (name, fn) in enumerate(SOURCES):
        if i in DEAD_SOURCES:
            continue
        try:
            df = fn(sym)
            time.sleep(0.15)
            if len(df) >= MIN_CANDLES:
                return name, df
        except Blocked as e:
            log(f"  {name} (Quelle {i}) gesperrt ({e}) – wird übersprungen")
            DEAD_SOURCES.add(i)
        except (NotFound, RuntimeError, ValueError, KeyError, TypeError):
            pass
    return None, None


# ----------------------------------------------------------------- Indikatoren
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def sma(s, n):
    return s.rolling(n).mean()


def lin(x, at100, at0):
    """Linear: Wert 'at100' -> 100 Punkte, Wert 'at0' -> 0 Punkte (begrenzt)."""
    return (100 * (x - at0) / (at100 - at0)).clip(0, 100)


def tanh_score(x):
    return 50 + 50 * np.tanh(x)


def supertrend(h, l, c, atr, mult=3.0):
    h, l, c, atr = h.values, l.values, c.values, atr.values
    n = len(c)
    up, dn = (h + l) / 2 - mult * atr, (h + l) / 2 + mult * atr
    fu, fd = np.full(n, np.nan), np.full(n, np.nan)
    d = np.zeros(n)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if i == 0 or np.isnan(fu[i - 1]):
            fu[i], fd[i], d[i] = up[i], dn[i], 1
            continue
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        if d[i - 1] == -1 and c[i] > fd[i - 1]:
            d[i] = 1
        elif d[i - 1] == 1 and c[i] < fu[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    line = np.where(d == 1, fu, fd)
    d[np.isnan(atr)] = np.nan
    return pd.Series(d), pd.Series(line)


def fmt(x, d=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def analyse(df):
    """Gibt (Score-Serie, Liste mit Indikator-Details der letzten Kerze, Kategorie-Scores) zurück."""
    o, h, l, c, v = df["o"], df["h"], df["l"], df["c"], df["v"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)
    atr_safe = atr.replace(0, np.nan)

    # Momentum
    delta = c.diff()
    rs = rma(delta.clip(lower=0), 14) / rma((-delta).clip(lower=0), 14).replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(100)
    rsi = rsi.where(rma(delta.abs(), 14).notna())
    rmin, rmax = rsi.rolling(14).min(), rsi.rolling(14).max()
    stoch = ((rsi - rmin) / (rmax - rmin).replace(0, np.nan)).fillna(0.5) * 100
    stoch = stoch.where(rmin.notna())
    k = sma(stoch, 3)
    dline = sma(k, 3)
    macd = ema(c, 12) - ema(c, 26)
    macd_sig = ema(macd, 9)
    hist = macd - macd_sig
    tp = (h + l + c) / 3
    tp_sma = sma(tp, 20)
    mad = tp.rolling(20).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
    cci = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))
    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    wr = -100 * (hh14 - c) / (hh14 - ll14).replace(0, np.nan)
    roc = (c / c.shift(12) - 1) * 100
    roc_z = (roc / 100) / ((atr_safe / c) * math.sqrt(12))

    # Trend
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    s50, s200 = sma(c, 50), sma(c, 200)
    up_m, dn_m = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0))
    mdm = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0))
    pdi = 100 * rma(pdm, 14) / atr_safe
    mdi = 100 * rma(mdm, 14) / atr_safe
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = rma(dx, 14)
    di_dir = (pdi - mdi) / (pdi + mdi).replace(0, np.nan)

    # Volumen
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_dev = (obv - ema(obv, 20)) / (obv.diff().abs().rolling(20).mean() * 4).replace(0, np.nan)
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    mfi = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).fillna(100).where(pos.notna())
    rng = (h - l).replace(0, np.nan)
    mfm = (((c - l) - (h - c)) / rng).fillna(0)
    cmf = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # Volatilität
    bb_mid = sma(c, 20)
    bb_std = c.rolling(20).std(ddof=0)
    pctb = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    kc_pos = (c - e20) / (2 * atr_safe)
    st_dir, st_line = supertrend(h, l, c, atr, 3.0)
    st_dist = (c - st_line).abs() / atr_safe

    # (Name, Kategorie, Teil-Score 0–100, Wert-Text, Erklärung)
    ind = [
        ("RSI (14)", "Momentum", lin(rsi, 30, 70), rsi.map(lambda x: fmt(x)),
         rsi.map(lambda x: "überverkauft" if x < 30 else "überkauft" if x > 70 else "neutraler Bereich")),
        ("Stochastic RSI", "Momentum", (lin(k, 20, 80) + np.where(k > dline, 10, -10)).clip(0, 100),
         k.map(lambda x: fmt(x)),
         k.map(lambda x: "tief – Erholung möglich" if x < 20 else "hoch – Rücksetzer möglich" if x > 80 else "Mitte")),
        ("MACD", "Momentum", tanh_score(hist / (0.25 * atr_safe)), hist.map(lambda x: fmt(x, 4)),
         hist.map(lambda x: "über Signallinie" if x > 0 else "unter Signallinie")),
        ("CCI (20)", "Momentum", lin(cci, -200, 200), cci.map(lambda x: fmt(x, 0)),
         cci.map(lambda x: "stark unter Durchschnitt" if x < -100 else "stark über Durchschnitt" if x > 100 else "normal")),
        ("Williams %R", "Momentum", lin(wr, -80, -20), wr.map(lambda x: fmt(x)),
         wr.map(lambda x: "überverkauft" if x < -80 else "überkauft" if x > -20 else "neutral")),
        ("ROC (12)", "Momentum", tanh_score(roc_z / 1.2), roc.map(lambda x: fmt(x, 2) + " %"),
         roc.map(lambda x: "Kurs steigt" if x > 0 else "Kurs fällt")),
        ("Preis vs. EMA 20", "Trend", tanh_score((c - e20) / (1.5 * atr_safe)), e20.map(lambda x: f"{x:.6g}"),
         (c - e20).map(lambda x: "Kurs über EMA 20" if x > 0 else "Kurs unter EMA 20")),
        ("EMA 20 / 50", "Trend", tanh_score((e20 - e50) / (2 * atr_safe)), (e20 - e50).map(lambda x: f"{x:.4g}"),
         (e20 - e50).map(lambda x: "kurzfristiger Aufwärtstrend" if x > 0 else "kurzfristiger Abwärtstrend")),
        ("EMA 50 / 200", "Trend", tanh_score((e50 - e200) / (4 * atr_safe)), (e50 - e200).map(lambda x: f"{x:.4g}"),
         (e50 - e200).map(lambda x: "EMA 50 über 200" if x > 0 else "EMA 50 unter 200")),
        ("SMA 50 / 200", "Trend", tanh_score((s50 - s200) / (4 * atr_safe)), (s50 - s200).map(lambda x: f"{x:.4g}"),
         (s50 - s200).map(lambda x: "Golden Cross" if x > 0 else "Death Cross")),
        ("Preis vs. SMA 200", "Trend", tanh_score((c - s200) / (5 * atr_safe)), s200.map(lambda x: f"{x:.6g}"),
         (c - s200).map(lambda x: "über langfristigem Schnitt" if x > 0 else "unter langfristigem Schnitt")),
        ("ADX / DMI", "Trend", 50 + 50 * np.tanh(2 * di_dir) * (adx / 30).clip(0, 1),
         adx.map(lambda x: fmt(x)),
         pd.Series([("starker " if a > 25 else "schwacher ") + ("Aufwärtstrend" if d > 0 else "Abwärtstrend")
                    if not (np.isnan(a) or np.isnan(d)) else "–" for a, d in zip(adx, di_dir)])),
        ("OBV", "Volumen", tanh_score(obv_dev), obv_dev.map(lambda x: fmt(x, 2)),
         obv_dev.map(lambda x: "Kaufvolumen überwiegt" if x > 0 else "Verkaufsvolumen überwiegt")),
        ("MFI (14)", "Volumen", lin(mfi, 20, 80), mfi.map(lambda x: fmt(x)),
         mfi.map(lambda x: "überverkauft" if x < 20 else "überkauft" if x > 80 else "neutral")),
        ("Chaikin Money Flow", "Volumen", lin(cmf, 0.25, -0.25), cmf.map(lambda x: fmt(x, 3)),
         cmf.map(lambda x: "Geld fließt hinein" if x > 0 else "Geld fließt ab")),
        ("Bollinger %B", "Volatilität", lin(pctb, 0, 1), pctb.map(lambda x: fmt(x, 2)),
         pctb.map(lambda x: "am unteren Band" if x < 0.1 else "am oberen Band" if x > 0.9 else "innerhalb der Bänder")),
        ("Keltner-Kanal", "Volatilität", lin(kc_pos, -1, 1), kc_pos.map(lambda x: fmt(x, 2)),
         kc_pos.map(lambda x: "unter dem Kanal" if x < -1 else "über dem Kanal" if x > 1 else "im Kanal")),
        ("Supertrend", "Volatilität", 50 + st_dir * (30 + 20 * np.tanh(st_dist / 2)),
         st_line.map(lambda x: f"{x:.6g}"),
         st_dir.map(lambda x: "Aufwärts-Signal" if x == 1 else "Abwärts-Signal" if x == -1 else "–")),
    ]

    subs = pd.DataFrame({name: pd.Series(np.asarray(s, dtype=float)) for name, _, s, _, _ in ind})
    cats = {}
    for cat in CATEGORY_WEIGHTS:
        cols = [name for name, ccat, *_ in ind if ccat == cat]
        cats[cat] = subs[cols].mean(axis=1, skipna=True)
    cat_df = pd.DataFrame(cats)
    w = pd.Series(CATEGORY_WEIGHTS)
    mask = cat_df.notna()
    raw = (cat_df.fillna(0) * w).sum(axis=1) / (mask * w).sum(axis=1).replace(0, np.nan)
    score = (50 + (raw - 50) * STRETCH).clip(0, 100)

    last = len(df) - 1
    details = []
    for name, cat, s, val, note in ind:
        sv = subs[name].iloc[last]
        if np.isnan(sv):
            continue
        details.append([name, cat, round(float(sv)), str(val.iloc[last]), str(note.iloc[last])])
    cat_last = {k: (None if np.isnan(x) else round(float(x))) for k, x in cat_df.iloc[last].items()}
    return score, details, cat_last


# ----------------------------------------------------------------- Ausgabe
def sig(x, digits=6):
    if x is None or not np.isfinite(x) or x == 0:
        return 0
    return float(f"{x:.{digits}g}")


def build_coin(meta, src, df):
    score, details, cats = analyse(df)
    tail = df.iloc[-HISTORY:]
    sc = score.iloc[-HISTORY:]
    candles = [[int(r.t), sig(r.o), sig(r.h), sig(r.l), sig(r.c)] for r in tail.itertuples()]
    scores = [None if np.isnan(x) else round(float(x), 1) for x in sc]
    current = next((x for x in reversed(scores) if x is not None), 50)
    return {
        "s": meta["sym"], "n": meta["name"], "r": meta["rank"],
        "p": sig(meta["price"] or float(tail["c"].iloc[-1])),
        "ch": None if meta["chg"] is None else round(float(meta["chg"]), 2),
        "mc": meta["mcap"], "src": src, "sc": current, "c": candles, "z": scores,
        "b": details, "k": cats,
    }


def main():
    started = time.time()
    coins = load_coins()
    out, seen = [], set()
    for meta in coins:
        if len(out) >= TOP_N:
            break
        sym = meta["sym"]
        if not sym or sym in seen:
            continue
        src, df = load_candles(sym)
        if df is None:
            log(f"  {sym}: keine 4h-Kerzen gefunden – übersprungen")
            continue
        try:
            out.append(build_coin(meta, src, df))
            seen.add(sym)
            log(f"  {len(out):3d}. {sym:<8} Score {out[-1]['sc']:5.1f}  ({src})")
        except Exception as e:
            log(f"  {sym}: Berechnung fehlgeschlagen: {e}")
    if not out:
        raise SystemExit("Keine Daten – Seite wird nicht überschrieben.")
    out.sort(key=lambda c: c["sc"], reverse=True)
    payload = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "buy": BUY_LEVEL, "sell": SELL_LEVEL, "weights": CATEGORY_WEIGHTS, "coins": out,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", data))
    log(f"Fertig: {len(out)} Coins in {time.time() - started:.0f}s -> {OUT_FILE}")


HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0e14">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Krypto Score Dashboard</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='22' fill='%230b0e14'/><path d='M18 70 L40 48 L55 60 L82 30' stroke='%2322c55e' stroke-width='10' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>">
<style>
:root{--bg:#0b0e14;--card:#131821;--card2:#1a202c;--line:#242b38;--txt:#e6e9ef;--mut:#8b93a5;
--buy:#22c55e;--sell:#ef4444;--neu:#eab308;--acc:#60a5fa}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif}
body{padding-bottom:env(safe-area-inset-bottom)}
header{position:sticky;top:0;z-index:5;background:rgba(11,14,20,.92);backdrop-filter:blur(10px);
padding:calc(10px + env(safe-area-inset-top)) 14px 10px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0 0 2px;font-weight:700;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:12px}
.search{margin-top:10px;width:100%;padding:11px 14px;border-radius:12px;border:1px solid var(--line);
background:var(--card);color:var(--txt);font-size:16px;outline:none}
.search:focus{border-color:var(--acc)}
.chips{display:flex;gap:8px;margin-top:10px;overflow-x:auto;scrollbar-width:none}
.chips::-webkit-scrollbar{display:none}
.chip{flex:none;padding:7px 12px;border-radius:999px;border:1px solid var(--line);background:var(--card);
color:var(--txt);font-size:13px;font-weight:600}
.chip.on{background:var(--txt);color:var(--bg);border-color:var(--txt)}
.chip.b.on{background:var(--buy);border-color:var(--buy);color:#04130a}
.chip.s.on{background:var(--sell);border-color:var(--sell);color:#fff}
.stats{display:flex;gap:8px;padding:12px 14px 4px}
.stat{flex:1;background:var(--card);border-radius:12px;padding:8px 10px}
.stat b{display:block;font-size:18px}.stat span{font-size:11px;color:var(--mut)}
ul{list-style:none;margin:0;padding:6px 10px 30px}
li.row{display:grid;grid-template-columns:28px 1fr auto;gap:10px;align-items:center;background:var(--card);
border-radius:14px;padding:11px 12px;margin:6px 0;cursor:pointer;transition:background .15s}
li.row:active{background:var(--card2)}
.rk{color:var(--mut);font-size:12px;text-align:center}
.nm{min-width:0}.nm b{font-size:15px}.nm small{color:var(--mut);margin-left:6px;font-size:12px}
.nm .px{font-size:12px;color:var(--mut);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar{height:6px;border-radius:3px;background:var(--line);margin-top:7px;position:relative;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.sc{min-width:52px;text-align:center;font-weight:800;font-size:17px;padding:6px 0;border-radius:10px}
.up{color:var(--buy)}.dn{color:var(--sell)}
.empty{color:var(--mut);text-align:center;padding:40px 0}
footer{color:var(--mut);font-size:11px;text-align:center;padding:0 20px 30px}
/* Detail */
#detail{position:fixed;inset:0;z-index:20;background:var(--bg);overflow-y:auto;display:none;
padding-bottom:calc(30px + env(safe-area-inset-bottom))}
#detail.open{display:block}
.dh{position:sticky;top:0;z-index:3;display:flex;align-items:center;gap:10px;background:rgba(11,14,20,.95);
backdrop-filter:blur(10px);padding:calc(10px + env(safe-area-inset-top)) 12px 10px;border-bottom:1px solid var(--line)}
.back{border:0;background:var(--card);color:var(--txt);width:38px;height:38px;border-radius:12px;font-size:20px}
.dh .t{flex:1;min-width:0}.dh .t b{font-size:17px}.dh .t div{font-size:12px;color:var(--mut)}
.box{margin:10px;background:var(--card);border-radius:16px;padding:10px 6px 6px;position:relative}
.box h3{font-size:13px;margin:2px 8px 6px;color:var(--mut);font-weight:600;display:flex;justify-content:space-between}
.legend{position:absolute;left:14px;top:30px;z-index:2;font-size:12px;color:var(--mut);pointer-events:none}
#pc{height:300px}#zc{height:170px}
.sec{margin:10px;background:var(--card);border-radius:16px;padding:12px}
.sec h4{margin:0 0 8px;font-size:14px}
.cats{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.cat{background:var(--card2);border-radius:10px;padding:8px 10px;font-size:12px;color:var(--mut)}
.cat b{display:block;font-size:17px;color:var(--txt)}
.pc2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.pc2 h5{margin:0 0 6px;font-size:13px}
.it{font-size:12px;background:var(--card2);border-radius:10px;padding:7px 9px;margin-bottom:6px}
.it b{font-size:13px;display:flex;justify-content:space-between}.it span{color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td{padding:7px 4px;border-top:1px solid var(--line);vertical-align:middle}
td.v{color:var(--mut);text-align:right;white-space:nowrap}
.mini{width:56px;height:6px;border-radius:3px;background:var(--line);position:relative;display:inline-block}
.mini i{position:absolute;top:0;bottom:0;left:0;border-radius:3px}
details summary{cursor:pointer;font-weight:600;font-size:14px}
.note{font-size:11px;color:var(--mut);margin:8px 10px 0;line-height:1.5}
</style>
</head>
<body>
<header>
  <h1>Krypto Score Dashboard</h1>
  <div class="sub" id="upd"></div>
  <input class="search" id="q" type="search" placeholder="Coin suchen (z. B. BTC, Solana)…" autocomplete="off">
  <div class="chips" id="chips">
    <button class="chip on" data-f="all">Alle</button>
    <button class="chip b" data-f="buy">Starker Kauf &gt;80</button>
    <button class="chip" data-f="pos">Kauf 60–80</button>
    <button class="chip" data-f="neu">Neutral</button>
    <button class="chip" data-f="neg">Verkauf 20–40</button>
    <button class="chip s" data-f="sell">Starker Verkauf &lt;20</button>
  </div>
</header>
<div class="stats" id="stats"></div>
<ul id="list"></ul>
<footer>Score 100 = maximales Kaufsignal · 50 = neutral · 0 = maximales Verkaufssignal.<br>
Rein technische Auswertung von 4-Stunden-Kerzen – keine Anlageberatung.</footer>

<div id="detail">
  <div class="dh">
    <button class="back" id="back" aria-label="Zurück">‹</button>
    <div class="t"><b id="dt"></b><div id="ds"></div></div>
    <div class="sc" id="dsc"></div>
  </div>
  <div class="box"><h3><span>Kurs (4h-Kerzen)</span><span>▲ Kauf ≥80 · ▼ Verkauf ≤20</span></h3>
    <div class="legend" id="pl"></div><div id="pc"></div></div>
  <div class="box"><h3><span>Score 0–100</span><span id="zl"></span></h3><div id="zc"></div></div>
  <div class="sec"><h4>Kategorien</h4><div class="cats" id="cats"></div></div>
  <div class="sec"><div class="pc2">
    <div><h5 class="up">Spricht für Kaufen</h5><div id="pro"></div></div>
    <div><h5 class="dn">Spricht für Verkaufen</h5><div id="con"></div></div>
  </div></div>
  <div class="sec"><details><summary>Alle Indikatoren</summary><table id="all"></table></details></div>
  <p class="note">Pfeile markieren die erste Kerze, mit der der Score in die Kauf- bzw. Verkaufszone eintritt.
  Datenquelle Kerzen: <span id="src"></span>. Keine Anlageberatung.</p>
</div>

<script>window.DATA = __DATA__;</script>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
 onerror="var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js';document.head.appendChild(s)"></script>
<script>
(function(){
const D = window.DATA, BUY = D.buy, SELL = D.sell;
const $ = id => document.getElementById(id);
let filter = 'all', query = '';

function color(s){
  if (s >= BUY) return '#22c55e';
  if (s >= 60) return '#84cc16';
  if (s > 40) return '#eab308';
  if (s > SELL) return '#f97316';
  return '#ef4444';
}
function label(s){
  if (s >= BUY) return 'Starker Kauf';
  if (s >= 60) return 'Kauf';
  if (s > 40) return 'Neutral';
  if (s > SELL) return 'Verkauf';
  return 'Starker Verkauf';
}
function price(p){
  if (p == null) return '–';
  const d = p >= 1000 ? 0 : p >= 1 ? 2 : p >= 0.01 ? 4 : 8;
  return '$' + p.toLocaleString('de-DE', {minimumFractionDigits: d, maximumFractionDigits: d});
}
function pct(x){ return x == null ? '' : (x >= 0 ? '+' : '') + x.toFixed(2).replace('.', ',') + ' %'; }
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

const upd = new Date(D.updated);
$('upd').textContent = 'Stand: ' + upd.toLocaleString('de-DE', {dateStyle:'medium', timeStyle:'short'}) +
  ' · ' + D.coins.length + ' Coins · Update alle 4 Std.';

const cnt = {buy:0, sell:0};
D.coins.forEach(c => { if (c.sc >= BUY) cnt.buy++; if (c.sc <= SELL) cnt.sell++; });
const avg = D.coins.reduce((a, c) => a + c.sc, 0) / D.coins.length;
$('stats').innerHTML =
  `<div class="stat"><b style="color:${color(avg)}">${avg.toFixed(0)}</b><span>Ø Markt-Score</span></div>` +
  `<div class="stat"><b class="up">${cnt.buy}</b><span>Starker Kauf</span></div>` +
  `<div class="stat"><b class="dn">${cnt.sell}</b><span>Starker Verkauf</span></div>`;

function match(c){
  if (query && !(c.s.toLowerCase().includes(query) || c.n.toLowerCase().includes(query))) return false;
  switch (filter){
    case 'buy': return c.sc >= BUY;
    case 'pos': return c.sc >= 60 && c.sc < BUY;
    case 'neu': return c.sc > 40 && c.sc < 60;
    case 'neg': return c.sc > SELL && c.sc <= 40;
    case 'sell': return c.sc <= SELL;
  }
  return true;
}
function render(){
  const rows = D.coins.filter(match);
  $('list').innerHTML = rows.length ? rows.map(c => {
    const col = color(c.sc);
    return `<li class="row" data-s="${esc(c.s)}">
      <div class="rk">#${c.r}</div>
      <div class="nm"><b>${esc(c.s)}</b><small>${esc(c.n)}</small>
        <div class="px">${price(c.p)} <span class="${c.ch >= 0 ? 'up' : 'dn'}">${pct(c.ch)}</span> · ${label(c.sc)}</div>
        <div class="bar"><i style="width:${c.sc}%;background:${col}"></i></div></div>
      <div class="sc" style="color:${col};background:${col}1f">${Math.round(c.sc)}</div></li>`;
  }).join('') : '<div class="empty">Keine Coins in diesem Filter.</div>';
}
$('q').addEventListener('input', e => { query = e.target.value.trim().toLowerCase(); render(); });
$('chips').addEventListener('click', e => {
  const b = e.target.closest('.chip'); if (!b) return;
  document.querySelectorAll('.chip').forEach(x => x.classList.toggle('on', x === b));
  filter = b.dataset.f; render();
});
$('list').addEventListener('click', e => {
  const li = e.target.closest('li.row'); if (li) { history.pushState({c: li.dataset.s}, '', '#' + li.dataset.s); open(li.dataset.s); }
});
$('back').addEventListener('click', () => history.length > 1 && location.hash ? history.back() : close());
window.addEventListener('popstate', () => { const s = location.hash.slice(1); s ? open(decodeURIComponent(s)) : close(); });
render();

let charts = [];
function close(){
  charts.forEach(c => c.remove()); charts = [];
  $('detail').classList.remove('open'); document.body.style.overflow = '';
}
function waitLib(cb, n){ if (window.LightweightCharts) cb(); else if ((n||0) < 100) setTimeout(() => waitLib(cb, (n||0)+1), 100); }

function open(sym){
  const c = D.coins.find(x => x.s === sym); if (!c) return close();
  close();
  $('detail').classList.add('open'); $('detail').scrollTop = 0; document.body.style.overflow = 'hidden';
  const col = color(c.sc);
  $('dt').textContent = c.s + ' · ' + c.n;
  $('ds').innerHTML = price(c.p) + ' <span class="' + (c.ch >= 0 ? 'up' : 'dn') + '">' + pct(c.ch) + '</span> · Rang #' + c.r;
  $('dsc').textContent = Math.round(c.sc); $('dsc').style.color = col; $('dsc').style.background = col + '1f';
  $('zl').textContent = label(c.sc);
  $('src').textContent = c.src;

  const order = ['Trend', 'Momentum', 'Volumen', 'Volatilität'];
  $('cats').innerHTML = order.map(k => {
    const v = c.k[k]; if (v == null) return '';
    return `<div class="cat">${k} <small>(${Math.round(D.weights[k]*100)} %)</small><b style="color:${color(v)}">${v}</b>
      <div class="bar"><i style="width:${v}%;background:${color(v)}"></i></div></div>`;
  }).join('');
  const sorted = [...c.b].sort((a, b) => b[2] - a[2]);
  const item = x => `<div class="it"><b><span style="color:var(--txt)">${esc(x[0])}</span><span style="color:${color(x[2])}">${x[2]}</span></b><span>${esc(x[4])}</span></div>`;
  const pro = sorted.filter(x => x[2] >= 60).slice(0, 4), con = sorted.filter(x => x[2] <= 40).reverse().slice(0, 4);
  $('pro').innerHTML = pro.length ? pro.map(item).join('') : '<div class="it"><span>Keine deutlichen Kaufsignale</span></div>';
  $('con').innerHTML = con.length ? con.map(item).join('') : '<div class="it"><span>Keine deutlichen Verkaufssignale</span></div>';
  $('all').innerHTML = order.map(cat => c.b.filter(x => x[1] === cat).map(x =>
    `<tr><td>${esc(x[0])}<br><span style="color:var(--mut);font-size:11px">${esc(x[4])}</span></td>
     <td class="v">${esc(x[3])}</td>
     <td class="v"><span class="mini"><i style="width:${x[2]}%;background:${color(x[2])}"></i></span> <b style="color:${color(x[2])}">${x[2]}</b></td></tr>`
  ).join('')).join('');

  waitLib(() => draw(c));
}

function draw(c){
  const L = window.LightweightCharts;
  const base = {
    autoSize: true,
    layout: {background: {type: 'solid', color: '#131821'}, textColor: '#8b93a5', fontSize: 11},
    grid: {vertLines: {color: '#1c2230'}, horzLines: {color: '#1c2230'}},
    rightPriceScale: {borderColor: '#242b38', minimumWidth: 64},
    timeScale: {borderColor: '#242b38', timeVisible: true, secondsVisible: false, rightOffset: 3},
    crosshair: {mode: L.CrosshairMode.Normal},
    localization: {locale: 'de-DE', timeFormatter: t => new Date(t * 1000).toLocaleString('de-DE', {day:'2-digit', month:'2-digit', hour:'2-digit', minute:'2-digit'})},
    handleScale: {axisPressedMouseMove: {time: true, price: false}},
  };
  const pc = L.createChart($('pc'), Object.assign({}, base, {timeScale: Object.assign({}, base.timeScale, {visible: false})}));
  const zc = L.createChart($('zc'), base);
  charts = [pc, zc];

  const last = c.c[c.c.length - 1][4];
  const prec = last >= 1000 ? 2 : last >= 1 ? 3 : last >= 0.01 ? 5 : 8;
  const cs = pc.addCandlestickSeries({upColor: '#22c55e', downColor: '#ef4444', borderVisible: false,
    wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    priceFormat: {type: 'price', precision: prec, minMove: Math.pow(10, -prec)}});
  cs.setData(c.c.map(k => ({time: k[0], open: k[1], high: k[2], low: k[3], close: k[4]})));

  const markers = [];
  for (let i = 0; i < c.z.length; i++){
    const s = c.z[i], p = i > 0 ? c.z[i-1] : null; if (s == null) continue;
    if (s >= BUY && (p == null || p < BUY)) markers.push({time: c.c[i][0], position: 'belowBar', color: '#22c55e', shape: 'arrowUp', text: 'Kauf'});
    if (s <= SELL && (p == null || p > SELL)) markers.push({time: c.c[i][0], position: 'aboveBar', color: '#ef4444', shape: 'arrowDown', text: 'Verkauf'});
  }
  cs.setMarkers(markers);

  const zs = zc.addBaselineSeries({baseValue: {type: 'price', price: 50}, lineWidth: 2,
    topLineColor: '#22c55e', topFillColor1: 'rgba(34,197,94,.28)', topFillColor2: 'rgba(34,197,94,.02)',
    bottomLineColor: '#ef4444', bottomFillColor1: 'rgba(239,68,68,.02)', bottomFillColor2: 'rgba(239,68,68,.28)',
    priceFormat: {type: 'price', precision: 0, minMove: 1},
    autoscaleInfoProvider: () => ({priceRange: {minValue: 0, maxValue: 100}})});
  const zdata = c.c.map((k, i) => c.z[i] == null ? {time: k[0]} : {time: k[0], value: c.z[i]});
  zs.setData(zdata);
  zs.createPriceLine({price: BUY, color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Kauf'});
  zs.createPriceLine({price: SELL, color: '#ef4444', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Verkauf'});
  zs.createPriceLine({price: 50, color: '#4b5563', lineWidth: 1, lineStyle: 3, axisLabelVisible: false});

  pc.timeScale().fitContent(); zc.timeScale().fitContent();

  // Zoom/Scroll synchronisieren
  let lock = false;
  const link = (a, b) => a.timeScale().subscribeVisibleLogicalRangeChange(r => {
    if (!r || lock) return; lock = true; b.timeScale().setVisibleLogicalRange(r); lock = false;
  });
  link(pc, zc); link(zc, pc);

  // Fadenkreuz synchronisieren + Legende
  const byTime = {}; c.c.forEach((k, i) => byTime[k[0]] = i);
  const legend = i => {
    const k = c.c[i], s = c.z[i];
    $('pl').innerHTML = `O ${k[1]} · H ${k[2]} · L ${k[3]} · C ${k[4]}` +
      (s != null ? ` · <b style="color:${color(s)}">Score ${Math.round(s)}</b>` : '');
  };
  legend(c.c.length - 1);
  pc.subscribeCrosshairMove(p => {
    if (lock) return;
    if (!p.time) { legend(c.c.length - 1); lock = true; zc.clearCrosshairPosition(); lock = false; return; }
    const i = byTime[p.time]; legend(i);
    lock = true; if (c.z[i] != null) zc.setCrosshairPosition(c.z[i], p.time, zs); lock = false;
  });
  zc.subscribeCrosshairMove(p => {
    if (lock) return;
    if (!p.time) { legend(c.c.length - 1); lock = true; pc.clearCrosshairPosition(); lock = false; return; }
    const i = byTime[p.time]; legend(i);
    lock = true; pc.setCrosshairPosition(c.c[i][4], p.time, cs); lock = false;
  });
}

const h = decodeURIComponent(location.hash.slice(1));
if (h) open(h);
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
