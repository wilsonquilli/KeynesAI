from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, session, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import time
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from supabase import create_client, Client
from dotenv import load_dotenv

try:
    from backend.stock import (
        DEFAULT_HORIZONS,
        add_features,
        add_horizon_features,
        add_pattern_features,
        build_main_model,
        download_data,
        get_main_predictors,
        predict_future,
        prepare_model_frame,
        run_for_stock,
    )
    from backend.boomCrash import BoomCrashModel
    from backend.stock_tree import StockTree
except ModuleNotFoundError:
    from stock import (
        DEFAULT_HORIZONS,
        add_features,
        add_horizon_features,
        add_pattern_features,
        build_main_model,
        download_data,
        get_main_predictors,
        predict_future,
        prepare_model_frame,
        run_for_stock,
    )
    from boomCrash import BoomCrashModel
    from stock_tree import StockTree

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = secrets.token_hex(32) 

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_KEY")
USERS_TABLE = os.environ.get("SUPABASE_USERS_TABLE", "users")
PORTFOLIO_ITEMS_TABLE = os.environ.get("SUPABASE_PORTFOLIO_TABLE", "portfolio_items")
FINNHUB_KEY = os.environ.get("FINNHUB_KEY")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
YAHOO_FINANCE_BASE_URL = "https://query1.finance.yahoo.com"
NASDAQ_BASE_URL = "https://api.nasdaq.com/api"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}
MARKET_REQUEST_TIMEOUT = float(os.environ.get("MARKET_REQUEST_TIMEOUT", "8"))
MARKET_SNAPSHOT_TTL_SECONDS = int(os.environ.get("MARKET_SNAPSHOT_TTL_SECONDS", "900"))
MARKET_MAX_WORKERS = max(4, min(12, int(os.environ.get("MARKET_MAX_WORKERS", "8"))))
FINNHUB_DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "NFLX", "PLTR", "JPM", "XOM", "AVGO", "IBM", "INTC", "CRM"
]
FINNHUB_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.environ.get("FINNHUB_SYMBOLS", ",".join(FINNHUB_DEFAULT_SYMBOLS)).split(",")
    if symbol.strip()
]
SYMBOL_PROFILE_OVERRIDES = {
    "AAPL": {"sector": "Technology"},
    "MSFT": {"sector": "Technology"},
    "NVDA": {"sector": "Technology"},
    "AMZN": {"sector": "Consumer Cyclical"},
    "GOOGL": {"sector": "Communication Services"},
    "META": {"sector": "Communication Services"},
    "TSLA": {"sector": "Consumer Cyclical"},
    "AMD": {"sector": "Technology"},
    "NFLX": {"sector": "Communication Services"},
    "PLTR": {"sector": "Technology"},
    "JPM": {"sector": "Financial Services"},
    "XOM": {"sector": "Energy"},
    "AVGO": {"sector": "Technology"},
    "IBM": {"sector": "Technology"},
    "INTC": {"sector": "Technology"},
    "CRM": {"sector": "Technology"},
}

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials. Check SUPABASE_URL and SUPABASE_SECRET_KEY/SUPABASE_KEY.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
_market_cache = {}
TICKER_PATTERN = re.compile(r"^[A-Z.\-^]{1,10}$")

def wants_json_response():
    accept_header = request.headers.get('Accept', '')
    return request.is_json or 'application/json' in accept_header

def get_request_value(key):
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        return (payload.get(key) or '').strip()
    return request.form.get(key, '').strip()

def json_error(message, status_code=400):
    return jsonify({'success': False, 'message': message}), status_code


def is_valid_ticker(value):
    return bool(TICKER_PATTERN.fullmatch(value or ""))

def format_supabase_error(err):
    if isinstance(err, dict):
        code = err.get("code")
        message = err.get("message", "Supabase error")
        if code == "PGRST205":
            return (
                "Supabase is connected, but the required table does not exist yet. "
                "Run backend/supabase_schema.sql in the Supabase SQL editor to create the users "
                "and portfolio_items tables."
            )
        return message

    error_message = str(err)
    if "PGRST205" in error_message or "schema cache" in error_message:
        return (
            "Supabase is connected, but the required table does not exist yet. "
            "Run backend/supabase_schema.sql in the Supabase SQL editor to create the users "
            "and portfolio_items tables."
        )
    return error_message

def format_finnhub_error(err):
    return f"Market data error: {str(err)}"

def require_logged_in_user():
    user_id = session.get('user_id')
    if not user_id:
        return None, json_error('You must be logged in to access this resource.', 401)
    return user_id, None

def get_user_by_email(email):
    response = (
        supabase.table(USERS_TABLE)
        .select("id,email,password")
        .eq("email", email)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None

def create_user(email, hashed_password):
    response = (
        supabase.table(USERS_TABLE)
        .insert({"email": email, "password": hashed_password})
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None

def serialize_portfolio_item(item):
    return {
        "id": item["id"],
        "sym": item["symbol"],
        "name": item["company_name"] or item["symbol"],
        "shares": item["shares"],
        "avg": item["avg_price"],
        "targetPrice": item["target_price"],
        "note": item["note"] or "",
        "price": "—",
        "chg": "—",
    }

def get_portfolio_items_for_user(user_id):
    response = (
        supabase.table(PORTFOLIO_ITEMS_TABLE)
        .select("id,category,symbol,company_name,shares,avg_price,target_price,note")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .execute()
    )

    grouped = {"owned": [], "watchlist": [], "wishlist": []}
    for item in response.data or []:
        category = item.get("category")
        if category in grouped:
            grouped[category].append(serialize_portfolio_item(item))
    return grouped

def create_portfolio_item(user_id, payload):
    category = payload.get("category")
    if category not in {"owned", "watchlist", "wishlist"}:
        raise ValueError("Invalid category.")

    symbol = (payload.get("sym") or "").strip().upper()
    company_name = (payload.get("name") or "").strip()

    if not symbol:
        raise ValueError("Ticker symbol is required.")

    insert_payload = {
        "user_id": user_id,
        "category": category,
        "symbol": symbol,
        "company_name": company_name or symbol,
        "shares": None,
        "avg_price": None,
        "target_price": None,
        "note": (payload.get("note") or "").strip() or None,
    }

    if category == "owned":
        insert_payload["shares"] = float(payload.get("shares") or 0)
        insert_payload["avg_price"] = float(payload.get("avg") or 0)
    if category == "wishlist":
        target_price = payload.get("targetPrice")
        insert_payload["target_price"] = float(target_price) if str(target_price).strip() else None

    response = supabase.table(PORTFOLIO_ITEMS_TABLE).insert(insert_payload).execute()
    rows = response.data or []
    if not rows:
        raise RuntimeError("Supabase did not return the created portfolio item.")
    return serialize_portfolio_item(rows[0] | {"category": category})

def delete_portfolio_item(user_id, item_id):
    response = (
        supabase.table(PORTFOLIO_ITEMS_TABLE)
        .delete()
        .eq("id", item_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(response.data)

def has_finnhub():
    return bool(FINNHUB_KEY)

def finnhub_get(path, params=None):
    if not has_finnhub():
        raise RuntimeError("Missing FINNHUB_KEY in backend/.env.")

    response = requests.get(
        f"{FINNHUB_BASE_URL}/{path}",
        params={**(params or {}), "token": FINNHUB_KEY},
        headers=HTTP_HEADERS,
        timeout=MARKET_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()

def yahoo_finance_get(path, params=None):
    response = requests.get(
        f"{YAHOO_FINANCE_BASE_URL}/{path}",
        params=params or {},
        headers=HTTP_HEADERS,
        timeout=MARKET_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()

def nasdaq_get(path, params=None):
    response = requests.get(
        f"{NASDAQ_BASE_URL}/{path}",
        params=params or {},
        headers={
            **HTTP_HEADERS,
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        },
        timeout=MARKET_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    status = payload.get("status") or {}
    if status.get("rCode") not in (None, 200):
        errors = status.get("bCodeMessage") or []
        error_text = "; ".join(item.get("errorMessage", "") for item in errors if item.get("errorMessage"))
        raise RuntimeError(error_text or f"Nasdaq API error for {path}.")
    return payload

def parse_money_text(value):
    cleaned = str(value or "").replace("$", "").replace(",", "").strip()
    return float(cleaned) if cleaned else 0.0

def parse_int_text(value):
    cleaned = str(value or "").replace(",", "").strip()
    return int(cleaned) if cleaned else 0

def get_cached_market_data(cache_key, ttl_seconds, loader):
    now = time.time()
    cached_entry = _market_cache.get(cache_key)
    if cached_entry and now - cached_entry["timestamp"] < ttl_seconds:
        return cached_entry["data"]

    try:
        data = loader()
    except Exception:
        if cached_entry:
            stale_data = dict(cached_entry["data"])
            stale_data["stale"] = True
            return stale_data
        raise

    _market_cache[cache_key] = {
        "timestamp": now,
        "data": data,
    }
    return data

def get_symbol_metrics(symbol):
    quote = finnhub_get("quote", {"symbol": symbol})
    profile = finnhub_get("stock/profile2", {"symbol": symbol})

    to_timestamp = int(time.time())
    from_timestamp = to_timestamp - (60 * 60 * 24 * 120)
    candles = finnhub_get(
        "stock/candle",
        {
            "symbol": symbol,
            "resolution": "D",
            "from": from_timestamp,
            "to": to_timestamp,
        },
    )

    closes = candles.get("c") or []
    volumes = candles.get("v") or []
    opens = candles.get("o") or []

    if candles.get("s") != "ok" or len(closes) < 31:
        raise RuntimeError(f"Not enough candle data returned for {symbol}.")

    current_price = float(quote.get("c") or closes[-1])
    previous_close = float(quote.get("pc") or closes[-2])
    day_change_percent = float(quote.get("dp") or 0)
    volume = volumes[-1] if volumes else 0
    open_price = float(opens[-1] or current_price)

    close_7 = float(closes[-7])
    close_14 = float(closes[-14])
    close_30 = float(closes[-30])
    avg_5 = sum(closes[-5:]) / 5
    avg_20 = sum(closes[-20:]) / 20

    momentum_7 = ((current_price - close_7) / close_7) * 100 if close_7 else 0
    momentum_14 = ((current_price - close_14) / close_14) * 100 if close_14 else 0
    momentum_30 = ((current_price - close_30) / close_30) * 100 if close_30 else 0

    score = (momentum_7 * 0.45) + (momentum_14 * 0.35) + (momentum_30 * 0.20)
    if current_price > avg_20:
        score += 2.5
    if avg_5 > avg_20:
        score += 2.5
    if day_change_percent > 0:
        score += 1

    direction = "up" if score >= 0 else "down"
    confidence = max(55, min(95, int(60 + abs(score) * 2.2)))
    horizon = "7d" if abs(momentum_7) >= abs(momentum_14) and abs(momentum_7) >= abs(momentum_30) else (
        "14d" if abs(momentum_14) >= abs(momentum_30) else "30d"
    )

    horizon_change = {
        "7d": momentum_7,
        "14d": momentum_14,
        "30d": momentum_30,
    }[horizon]

    if direction == "up":
        target_price = current_price * (1 + max(horizon_change, 1.5) / 100)
    else:
        target_price = current_price * (1 - max(abs(horizon_change), 1.5) / 100)

    return {
        "sym": symbol,
        "name": profile.get("name") or symbol,
        "sector": profile.get("finnhubIndustry") or "Unknown",
        "price": f"{current_price:.2f}",
        "current": f"{current_price:.2f}",
        "target": f"{target_price:.2f}",
        "chg": f"{day_change_percent:+.2f}%",
        "change_percent": day_change_percent,
        "vol": f"{volume / 1_000_000:.1f}M" if volume else "—",
        "mktCap": f"{profile.get('marketCapitalization', 0) / 1000:.2f}T" if (profile.get("marketCapitalization") or 0) >= 1000 else (
            f"{profile.get('marketCapitalization', 0):.0f}B" if profile.get("marketCapitalization") else "—"
        ),
        "direction": direction,
        "confidence": confidence,
        "horizon": horizon,
        "rationale": (
            f"{symbol} is trading at ${current_price:.2f} with a {day_change_percent:+.2f}% move today. "
            f"Momentum is {momentum_7:+.1f}% over 7d, {momentum_14:+.1f}% over 14d, and {momentum_30:+.1f}% over 30d. "
            f"The 5-day average is {'above' if avg_5 > avg_20 else 'below'} the 20-day average, "
            f"which supports a {'bullish' if direction == 'up' else 'bearish'} signal."
        ),
        "reason": (
            f"{symbol} is down {day_change_percent:.2f}% today with {momentum_7:+.1f}% 7-day momentum "
            f"and a {'weaker' if avg_5 < avg_20 else 'mixed'} short-term trend."
        ),
        "raw_score": score,
        "open": open_price,
    }

def get_yahoo_quotes(symbols):
    payload = yahoo_finance_get(
        "v7/finance/quote",
        {"symbols": ",".join(symbols)},
    )
    results = payload.get("quoteResponse", {}).get("result", [])
    return {
        item.get("symbol"): item
        for item in results
        if item.get("symbol")
    }

def get_yahoo_symbol_metrics(symbol, quote_data=None):
    chart_payload = yahoo_finance_get(
        f"v8/finance/chart/{symbol}",
        {
            "range": "6mo",
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        },
    )

    chart_result = (chart_payload.get("chart", {}).get("result") or [None])[0]
    chart_error = chart_payload.get("chart", {}).get("error")

    if chart_error:
        raise RuntimeError(chart_error.get("description") or f"Yahoo Finance chart error for {symbol}.")
    if not chart_result:
        raise RuntimeError(f"No Yahoo Finance chart data returned for {symbol}.")

    indicators = chart_result.get("indicators", {}).get("quote", [])
    if not indicators:
        raise RuntimeError(f"Missing quote history for {symbol}.")

    quote_history = indicators[0]
    closes = [float(value) for value in (quote_history.get("close") or []) if value is not None]
    opens = [float(value) for value in (quote_history.get("open") or []) if value is not None]
    volumes = [int(value) for value in (quote_history.get("volume") or []) if value is not None]

    if len(closes) < 31:
        raise RuntimeError(f"Not enough historical data returned for {symbol}.")

    current_price = float(closes[-1])
    previous_close = float(closes[-2])
    day_change_percent = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
    volume = int(volumes[-1]) if volumes else 0
    open_price = float(opens[-1]) if opens and not pd.isna(opens[-1]) else current_price

    close_7 = float(closes[-7])
    close_14 = float(closes[-14])
    close_30 = float(closes[-30])
    avg_5 = sum(closes[-5:]) / 5
    avg_20 = sum(closes[-20:]) / 20

    momentum_7 = ((current_price - close_7) / close_7) * 100 if close_7 else 0
    momentum_14 = ((current_price - close_14) / close_14) * 100 if close_14 else 0
    momentum_30 = ((current_price - close_30) / close_30) * 100 if close_30 else 0

    score = (momentum_7 * 0.45) + (momentum_14 * 0.35) + (momentum_30 * 0.20)
    if current_price > avg_20:
        score += 2.5
    if avg_5 > avg_20:
        score += 2.5
    if day_change_percent > 0:
        score += 1

    direction = "up" if score >= 0 else "down"
    confidence = max(55, min(95, int(60 + abs(score) * 2.2)))
    horizon = "7d" if abs(momentum_7) >= abs(momentum_14) and abs(momentum_7) >= abs(momentum_30) else (
        "14d" if abs(momentum_14) >= abs(momentum_30) else "30d"
    )
    horizon_change = {"7d": momentum_7, "14d": momentum_14, "30d": momentum_30}[horizon]
    target_price = current_price * (1 + max(horizon_change, 1.5) / 100) if direction == "up" else current_price * (1 - max(abs(horizon_change), 1.5) / 100)

    meta = chart_result.get("meta") or {}
    quote_data = quote_data or {}

    company_name = (
        quote_data.get("longName")
        or quote_data.get("shortName")
        or meta.get("symbol")
        or symbol
    )
    sector = quote_data.get("sector") or quote_data.get("industryDisp") or "Unknown"
    market_cap = (
        quote_data.get("marketCap")
        or meta.get("marketCap")
        or 0
    )

    return {
        "sym": symbol,
        "name": company_name,
        "sector": sector,
        "price": f"{current_price:.2f}",
        "current": f"{current_price:.2f}",
        "target": f"{target_price:.2f}",
        "chg": f"{day_change_percent:+.2f}%",
        "change_percent": day_change_percent,
        "vol": f"{volume / 1_000_000:.1f}M" if volume else "—",
        "mktCap": f"{market_cap / 1_000_000_000_000:.2f}T" if market_cap >= 1_000_000_000_000 else (
            f"{market_cap / 1_000_000_000:.0f}B" if market_cap else "—"
        ),
        "direction": direction,
        "confidence": confidence,
        "horizon": horizon,
        "rationale": (
            f"{symbol} is trading at ${current_price:.2f} with a {day_change_percent:+.2f}% move today. "
            f"Momentum is {momentum_7:+.1f}% over 7d, {momentum_14:+.1f}% over 14d, and {momentum_30:+.1f}% over 30d. "
            f"The 5-day average is {'above' if avg_5 > avg_20 else 'below'} the 20-day average, "
            f"which supports a {'bullish' if direction == 'up' else 'bearish'} signal."
        ),
        "reason": (
            f"{symbol} is down {day_change_percent:.2f}% today with {momentum_7:+.1f}% 7-day momentum "
            f"and a {'weaker' if avg_5 < avg_20 else 'mixed'} short-term trend."
        ),
        "raw_score": score,
        "open": open_price,
    }

def get_nasdaq_symbol_metrics(symbol):
    start_date = time.strftime("%Y-%m-%d", time.localtime(time.time() - (60 * 60 * 24 * 220)))

    history_payload = nasdaq_get(
        f"quote/{symbol}/historical",
        {
            "assetclass": "stocks",
            "fromdate": start_date,
            "limit": "120",
        },
    )
    info_payload = nasdaq_get(
        f"quote/{symbol}/info",
        {"assetclass": "stocks"},
    )

    rows = (history_payload.get("data") or {}).get("tradesTable", {}).get("rows") or []
    if len(rows) < 31:
        raise RuntimeError(f"Not enough Nasdaq historical data returned for {symbol}.")

    ordered_rows = list(reversed(rows))
    closes = [parse_money_text(row.get("close")) for row in ordered_rows]
    opens = [parse_money_text(row.get("open")) for row in ordered_rows]
    volumes = [parse_int_text(row.get("volume")) for row in ordered_rows]

    current_price = float(closes[-1])
    previous_close = float(closes[-2])
    day_change_percent = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
    volume = int(volumes[-1]) if volumes else 0
    open_price = float(opens[-1]) if opens and not pd.isna(opens[-1]) else current_price

    close_7 = float(closes[-7])
    close_14 = float(closes[-14])
    close_30 = float(closes[-30])
    avg_5 = sum(closes[-5:]) / 5
    avg_20 = sum(closes[-20:]) / 20

    momentum_7 = ((current_price - close_7) / close_7) * 100 if close_7 else 0
    momentum_14 = ((current_price - close_14) / close_14) * 100 if close_14 else 0
    momentum_30 = ((current_price - close_30) / close_30) * 100 if close_30 else 0

    score = (momentum_7 * 0.45) + (momentum_14 * 0.35) + (momentum_30 * 0.20)
    if current_price > avg_20:
        score += 2.5
    if avg_5 > avg_20:
        score += 2.5
    if day_change_percent > 0:
        score += 1

    direction = "up" if score >= 0 else "down"
    confidence = max(55, min(95, int(60 + abs(score) * 2.2)))
    horizon = "7d" if abs(momentum_7) >= abs(momentum_14) and abs(momentum_7) >= abs(momentum_30) else (
        "14d" if abs(momentum_14) >= abs(momentum_30) else "30d"
    )
    horizon_change = {"7d": momentum_7, "14d": momentum_14, "30d": momentum_30}[horizon]
    target_price = current_price * (1 + max(horizon_change, 1.5) / 100) if direction == "up" else current_price * (1 - max(abs(horizon_change), 1.5) / 100)

    info = info_payload.get("data") or {}
    company_name = info.get("companyName") or symbol
    sector = SYMBOL_PROFILE_OVERRIDES.get(symbol, {}).get("sector", "Unknown")

    return {
        "sym": symbol,
        "name": company_name.replace(" Common Stock", ""),
        "sector": sector,
        "price": f"{current_price:.2f}",
        "current": f"{current_price:.2f}",
        "target": f"{target_price:.2f}",
        "chg": f"{day_change_percent:+.2f}%",
        "change_percent": day_change_percent,
        "vol": f"{volume / 1_000_000:.1f}M" if volume else "—",
        "mktCap": "—",
        "direction": direction,
        "confidence": confidence,
        "horizon": horizon,
        "rationale": (
            f"{symbol} is trading at ${current_price:.2f} with a {day_change_percent:+.2f}% move today. "
            f"Momentum is {momentum_7:+.1f}% over 7d, {momentum_14:+.1f}% over 14d, and {momentum_30:+.1f}% over 30d. "
            f"The 5-day average is {'above' if avg_5 > avg_20 else 'below'} the 20-day average, "
            f"which supports a {'bullish' if direction == 'up' else 'bearish'} signal."
        ),
        "reason": (
            f"{symbol} is down {day_change_percent:.2f}% today with {momentum_7:+.1f}% 7-day momentum "
            f"and a {'weaker' if avg_5 < avg_20 else 'mixed'} short-term trend."
        ),
        "raw_score": score,
        "open": open_price,
    }


def build_market_snapshot(metrics, failures, source):
    gainers = sorted(metrics, key=lambda item: item["change_percent"], reverse=True)[:8]
    losers = sorted(metrics, key=lambda item: item["change_percent"])[:8]
    predictions = sorted(metrics, key=lambda item: item["confidence"], reverse=True)

    return {
        "updated_at": int(time.time()),
        "symbols_analyzed": len(metrics),
        "failures": failures,
        "source": source,
        "gainers": gainers,
        "losers": losers,
        "predictions": predictions,
    }

def load_market_metrics(symbols, loader):
    metrics = []
    failures = []
    for symbol in symbols:
        try:
            metrics.append(loader(symbol))
        except Exception as err:
            failures.append({"symbol": symbol, "error": str(err)})
    return metrics, failures


def get_market_snapshot():
    def load_market_metric_for_symbol(symbol):
        finnhub_error = None
        nasdaq_error = None

        try:
            metric = get_nasdaq_symbol_metrics(symbol)
            metric["source"] = "Nasdaq"
            return metric, None
        except Exception as err:
            nasdaq_error = str(err)

        if has_finnhub():
            try:
                metric = get_symbol_metrics(symbol)
                metric["source"] = "Finnhub"
                return metric, None
            except Exception as err:
                finnhub_error = str(err)

        try:
            metric = get_yahoo_symbol_metrics(symbol)
            metric["source"] = "Yahoo Finance"
            return metric, None
        except Exception as err:
            failure = {"symbol": symbol, "error": str(err)}
            if nasdaq_error:
                failure["nasdaq_error"] = nasdaq_error
            if finnhub_error:
                failure["finnhub_error"] = finnhub_error
            return None, failure

    def load_snapshot():
        metrics = []
        failures = []
        source_counts = {"Finnhub": 0, "Nasdaq": 0, "Yahoo Finance": 0}

        with ThreadPoolExecutor(max_workers=min(MARKET_MAX_WORKERS, max(1, len(FINNHUB_SYMBOLS)))) as executor:
            future_map = {
                executor.submit(load_market_metric_for_symbol, symbol): symbol
                for symbol in FINNHUB_SYMBOLS
            }

            for future in as_completed(future_map):
                metric, failure = future.result()
                if metric:
                    metrics.append(metric)
                    source_counts[metric["source"]] += 1
                elif failure:
                    failures.append(failure)

        if metrics:
            active_sources = [name for name, count in source_counts.items() if count]
            if len(active_sources) > 1:
                source = " + ".join(active_sources)
            elif active_sources:
                source = active_sources[0]
            else:
                source = "Market Data"
            return build_market_snapshot(metrics, failures, source)

        if has_finnhub():
            raise RuntimeError("Unable to load market data from Finnhub, Nasdaq, or Yahoo Finance for any configured symbol.")
        raise RuntimeError("Unable to load market data from Nasdaq or Yahoo Finance for any configured symbol.")

    return get_cached_market_data("market_snapshot", MARKET_SNAPSHOT_TTL_SECONDS, load_snapshot)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = get_request_value('email')
        password = get_request_value('password')

        if not email or not password:
            if wants_json_response():
                return json_error('Email and password are required.')
            flash('Email and password are required.', 'error')
            return render_template('login.html')
        
        try:
            user = get_user_by_email(email)
            
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['user_email'] = user['email']
                if wants_json_response():
                    return jsonify({
                        'success': True,
                        'message': 'Login successful!',
                        'user': {
                            'id': user['id'],
                            'email': user['email']
                        }
                    })
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                if wants_json_response():
                    return json_error('Invalid email or password.', 401)
                flash('Invalid email or password', 'error')
        except Exception as e:
            if wants_json_response():
                return json_error(f'Database error: {format_supabase_error(e)}', 500)
            flash(f'Database error: {format_supabase_error(e)}', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = get_request_value('email')
        password = get_request_value('password')

        if not email or not password:
            if wants_json_response():
                return json_error('Email and password are required.')
            flash('Email and password are required.', 'error')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)
        
        print(f"Attempting to register user: {email}")
        
        try:
            existing_user = get_user_by_email(email)
            if existing_user:
                if wants_json_response():
                    return json_error('Email already registered.', 409)
                flash('Email already registered', 'error')
                return render_template('register.html')

            created_user = create_user(email, hashed_password)
            if not created_user:
                if wants_json_response():
                    return json_error('Failed to create user in Supabase.', 500)
                flash('Failed to create user in Supabase.', 'error')
                return render_template('register.html')

            user_id = created_user['id']
            print("User successfully registered in Supabase")

            if wants_json_response():
                session['user_id'] = user_id
                session['user_email'] = email
                return jsonify({
                    'success': True,
                    'message': 'Registration successful!',
                    'user': {
                        'id': user_id,
                        'email': email
                    }
                }), 201

            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as err:
            print(f"Supabase error during registration: {err}")
            if wants_json_response():
                return json_error(f'Database error: {format_supabase_error(err)}', 500)
            flash(f'Database error: {format_supabase_error(err)}', 'error')
    
    return render_template('register.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('user_id', None)
    session.pop('user_email', None)
    if wants_json_response():
        return jsonify({'success': True, 'message': 'You have been logged out.'})
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/api/auth/status')
def auth_status():
    user_id = session.get('user_id')
    user_email = session.get('user_email')
    return jsonify({
        'logged_in': bool(user_id),
        'user': {
            'id': user_id,
            'email': user_email
        } if user_id else None
    })

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    user_id, auth_error = require_logged_in_user()
    if auth_error:
        return auth_error

    try:
        return jsonify({
            "success": True,
            "portfolio": get_portfolio_items_for_user(user_id)
        })
    except Exception as err:
        return json_error(f"Database error: {format_supabase_error(err)}", 500)

@app.route('/api/portfolio/items', methods=['POST'])
def add_portfolio_item():
    user_id, auth_error = require_logged_in_user()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True) or {}
    try:
        created_item = create_portfolio_item(user_id, payload)
        return jsonify({
            "success": True,
            "item": created_item,
            "category": payload.get("category")
        }), 201
    except ValueError as err:
        return json_error(str(err), 400)
    except Exception as err:
        return json_error(f"Database error: {format_supabase_error(err)}", 500)

@app.route('/api/portfolio/items/<item_id>', methods=['DELETE'])
def remove_portfolio_item(item_id):
    user_id, auth_error = require_logged_in_user()
    if auth_error:
        return auth_error

    try:
        deleted = delete_portfolio_item(user_id, item_id)
        if not deleted:
            return json_error("Portfolio item not found.", 404)
        return jsonify({"success": True, "message": "Portfolio item deleted."})
    except Exception as err:
        return json_error(f"Database error: {format_supabase_error(err)}", 500)

@app.route('/api/market/trending', methods=['GET'])
def market_trending():
    try:
        snapshot = get_market_snapshot()
        return jsonify({
            "success": True,
            "updated_at": snapshot["updated_at"],
            "symbols_analyzed": snapshot["symbols_analyzed"],
            "source": snapshot.get("source", "Finnhub"),
            "gainers": snapshot["gainers"],
            "losers": snapshot["losers"],
            "failures": snapshot["failures"],
        })
    except Exception as err:
        return json_error(format_finnhub_error(err), 500)

@app.route('/api/market/predictions', methods=['GET'])
def market_predictions():
    try:
        snapshot = get_market_snapshot()
        return jsonify({
            "success": True,
            "updated_at": snapshot["updated_at"],
            "symbols_analyzed": snapshot["symbols_analyzed"],
            "source": snapshot.get("source", "Finnhub"),
            "predictions": snapshot["predictions"],
            "failures": snapshot["failures"],
        })
    except Exception as err:
        return json_error(format_finnhub_error(err), 500)


stock_tree = StockTree()

@app.route('/sectors.html')
def sectors():
    category_path = request.args.get('category', '').split('/')
    if category_path == ['']:
        category_path = []
    category_info = stock_tree.get_category_info(category_path)
    stocks = stock_tree.get_stocks_in_category(category_path)
    return render_template(
        'sectors.html',
        category_info=category_info,
        current_path=category_path,
        stocks=stocks
    )

@app.route('/')
def home():
    return render_template('index.HTML')

@app.route('/about_us.html')
def about():  
    return render_template('about_us.html')

@app.route('/predictions.html')
def predictions():
    ticker = request.args.get('ticker', 'AAPL') 

    try:
        df = prepare_model_frame(download_data(ticker), DEFAULT_HORIZONS)
        predictors = get_main_predictors(DEFAULT_HORIZONS)
        model = build_main_model()
        model.fit(df[predictors], df["Target"])

        predictions_df = predict_future(df, model, predictors, ticker, horizons=DEFAULT_HORIZONS)

        if predictions_df is not None:
            table_html = predictions_df.head(20).to_html(classes="prediction-table", border=0)
        else:
            table_html = "<p style='color:red;'>No predictions returned.</p>"

    except Exception as e:
        table_html = f"<p style='color:red;'>Error generating predictions for {ticker}: {e}</p>"

    return render_template('predictions.html', predictions_table=table_html, selected_ticker=ticker)

@app.route('/portfolio.html')
def portfolio():
    portfolio_data = [
        {"name": "AAPL", "shares": 10, "buy_price": 150.00, "current_price": 165.32},
        {"name": "IBM", "shares": 5, "buy_price": 130.00, "current_price": 127.45},
        {"name": "MSFT", "shares": 8, "buy_price": 290.00, "current_price": 312.25},
        {"name": "S&P500", "shares": 20, "buy_price": 4000.00, "current_price": 4180.22},
    ]

    total_value = 0
    total_cost = 0
    top_stock = None
    best_gain = float('-inf')

    for stock in portfolio_data:
        stock["value"] = stock["shares"] * stock["current_price"]
        stock["cost"] = stock["shares"] * stock["buy_price"]
        stock["change_percent"] = round(((stock["current_price"] - stock["buy_price"]) / stock["buy_price"]) * 100, 2)
        total_value += stock["value"]
        total_cost += stock["cost"]
        if stock["change_percent"] > best_gain:
            best_gain = stock["change_percent"]
            top_stock = stock["name"]

    gain_loss = round(total_value - total_cost, 2)

    return render_template("portfolio.html",
                           portfolio=portfolio_data,
                           total_value=round(total_value, 2),
                           gain_loss=gain_loss,
                           top_stock=top_stock)

@app.route('/trending_stocks.html')
def trending_stocks():
    return render_template('trending_stocks.html')

@app.route('/nick.html')
def nick():
    return render_template('nick.html')

@app.route('/sam.html')
def sam():
    return render_template('sam.html')

@app.route('/wilson.html')
def wilson():
    return render_template('wilson.html')

@app.route('/mostafa.html')
def mostafa():
    return render_template('mostafa.html')

@app.route('/style.css')
def style():
    return send_from_directory('static', 'style.css')

if __name__ == '__main__':
    app.run(debug=True)
