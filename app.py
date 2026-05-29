"""
台股買賣訊號系統 — Streamlit App
套件: pip install streamlit yfinance google-generativeai requests supabase apscheduler python-dotenv
執行: streamlit run app.py
"""

import os, json, requests, threading, time
from datetime import datetime, timedelta

import streamlit as st
import yfinance as yf
import pandas as pd
from google import genai
from google.genai import types as genai_types
from supabase import create_client
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

load_dotenv()

# ── 設定 ────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY    = os.getenv("NEWS_API_KEY")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_ANON_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Supabase helpers ─────────────────────────────────────────
def db_get_watchlist():
    res = supabase.table("watchlist").select("*").order("added_at").execute()
    return res.data or []

def db_add_stock(ticker: str, name: str):
    supabase.table("watchlist").upsert({"ticker": ticker, "name": name}).execute()

def db_remove_stock(ticker: str):
    supabase.table("watchlist").delete().eq("ticker", ticker).execute()

def db_save_signal(data: dict):
    supabase.table("signals").insert(data).execute()

def db_get_signals(days: int = 30):
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    res = (supabase.table("signals")
           .select("*")
           .gte("analyzed_at", since)
           .order("analyzed_at", desc=True)
           .execute())
    return res.data or []

def db_get_latest_signals():
    """每支股票最新一筆訊號"""
    all_signals = db_get_signals(days=3)
    seen = {}
    for s in all_signals:
        if s["ticker"] not in seen:
            seen[s["ticker"]] = s
    return list(seen.values())

# ── 分析核心 ─────────────────────────────────────────────────
def get_technical_score(ticker: str) -> dict:
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if df.empty or len(df) < 60:
        return {"score": 0, "price": None, "detail": "股價資料不足"}

    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))

    ema12     = close.ewm(span=12).mean()
    ema26     = close.ewm(span=26).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    vol_ma20  = volume.rolling(20).mean()

    price      = round(float(close.iloc[-1]), 2)
    cur_ma20   = float(ma20.iloc[-1])
    cur_ma60   = float(ma60.iloc[-1])
    cur_rsi    = float(rsi.iloc[-1])
    cur_mh     = float(macd_hist.iloc[-1])
    cur_vol    = float(volume.iloc[-1])
    cur_vol_ma = float(vol_ma20.iloc[-1])

    score, signals = 0, []

    if price > cur_ma20 > cur_ma60:
        score += 1; signals.append("✅ 多頭排列（價 > MA20 > MA60）")
    elif price < cur_ma20 < cur_ma60:
        score -= 1; signals.append("❌ 空頭排列（價 < MA20 < MA60）")
    else:
        signals.append("➖ 均線糾結")

    if cur_rsi < 30:
        score += 1; signals.append(f"✅ RSI 超賣（{cur_rsi:.1f}）")
    elif cur_rsi > 70:
        score -= 1; signals.append(f"❌ RSI 超買（{cur_rsi:.1f}）")
    else:
        signals.append(f"➖ RSI 中性（{cur_rsi:.1f}）")

    if cur_mh > 0:
        score += 1; signals.append("✅ MACD 柱狀轉正")
    else:
        score -= 1; signals.append("❌ MACD 柱狀為負")

    if cur_vol > cur_vol_ma * 1.5:
        score += 1; signals.append("✅ 成交量放大（>1.5x 均量）")
    else:
        signals.append("➖ 成交量正常")

    return {
        "score":  score,
        "price":  price,
        "ma20":   round(cur_ma20, 2),
        "ma60":   round(cur_ma60, 2),
        "rsi":    round(cur_rsi, 1),
        "detail": "\n".join(signals),
    }

def get_fundamental_score(ticker: str) -> dict:
    """
    用 TWSE 證交所 API 抓台股基本面，避免 yfinance rate limit。
    ticker 格式: "2330.TW" -> stock_id: "2330"
    """
    stock_id = ticker.replace(".TW", "").replace(".TWO", "")
    pe, dy, rev_g = None, 0, None
    score, signals = 0, []
    headers = {"User-Agent": "Mozilla/5.0"}

    # PE、殖利率：TWSE 個股本益比
    try:
        url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?stockNo={stock_id}&response=json"
        r = requests.get(url, headers=headers, timeout=10)
        rows = r.json().get("data", [])
        if rows:
            latest = rows[-1]
            try:   dy = float(latest[1]) / 100
            except: dy = 0
            try:   pe = float(latest[3])
            except: pe = None
    except Exception as e:
        signals.append(f"➖ TWSE 抓取失敗：{e}")

    # EPS 季增率：TWSE 每季 EPS
    try:
        url2 = f"https://www.twse.com.tw/rwd/zh/finance/t163sb04?stockNo={stock_id}&response=json"
        r2 = requests.get(url2, headers=headers, timeout=10)
        rows2 = r2.json().get("data", [])
        if len(rows2) >= 2:
            eps_now  = float(rows2[-1][1])
            eps_prev = float(rows2[-2][1])
            if eps_prev != 0:
                rev_g = (eps_now - eps_prev) / abs(eps_prev)
    except Exception:
        pass

    # 評分
    if pe:
        if pe < 15:   score += 1; signals.append(f"✅ PE 偏低（{pe:.1f}x）")
        elif pe > 30: score -= 1; signals.append(f"❌ PE 偏高（{pe:.1f}x）")
        else:         signals.append(f"➖ PE 合理（{pe:.1f}x）")
    else:
        signals.append("➖ PE 無資料")

    if dy > 0.05:
        score += 1; signals.append(f"✅ 殖利率高（{dy*100:.1f}%）")
    elif dy > 0.02:
        signals.append(f"➖ 殖利率普通（{dy*100:.1f}%）")
    else:
        signals.append("➖ 殖利率無資料")

    if rev_g is not None:
        if rev_g > 0.1:  score += 1; signals.append(f"✅ EPS 季增 {rev_g*100:.1f}%")
        elif rev_g < 0:  score -= 1; signals.append(f"❌ EPS 季減 {rev_g*100:.1f}%")
        else:            signals.append(f"➖ EPS 持平（{rev_g*100:.1f}%）")
    return {
        "score":  score,
        "pe":     pe,
        "detail": "\n".join(signals) if signals else "基本面資料不足",
    }


def get_news_summary(ticker: str, name: str) -> str:
    items = []
    try:
        r = requests.get("https://newsapi.org/v2/everything", timeout=10, params={
            "q":        f"{name} OR {ticker.replace('.TW','')}",
            "language": "zh",
            "sortBy":   "publishedAt",
            "pageSize": 5,
            "from":     (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "apiKey":   NEWS_API_KEY,
        })
        for a in r.json().get("articles", []):
            items.append(f"[個股] {a['title']} — {(a.get('description') or '')[:80]}")
    except Exception as e:
        items.append(f"[NewsAPI 錯誤] {e}")

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                f"用繁體中文，搜尋近7天影響台灣股市或{name}({ticker})的重大事件"
                f"（地緣政治、Fed政策、半導體產業、台灣經濟數據等），列出最重要3條，每條一句話。"
            ),
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
            ),
        )
        items.append(f"[宏觀事件]\n{resp.text}")
    except Exception as e:
        items.append(f"[Gemini Search 錯誤] {e}")

    return "\n".join(items)

def get_gemini_signal(ticker, name, tech, fund, news) -> dict:
    prompt = f"""
你是專業台股分析師。根據以下資料判斷今日 {name}（{ticker}）的操作建議。

## 技術面（評分 {tech['score']}）
現價：{tech.get('price')} 元
{tech['detail']}

## 基本面（評分 {fund['score']}）
{fund['detail']}

## 市場新聞（近7天）
{news}

請輸出以下 JSON，不要有任何 markdown 包裝：
{{"signal":"BUY"或"HOLD"或"SELL","confidence":"高"或"中"或"低","reason":"繁體中文判斷理由含新聞影響，100字內","risk":"主要風險，50字內"}}
"""
    resp = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    try:
        raw = resp.text.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception:
        return {"signal": "HOLD", "confidence": "低", "reason": resp.text[:200], "risk": "解析失敗"}

def run_analysis(ticker: str, name: str) -> dict:
    tech  = get_technical_score(ticker)
    fund  = get_fundamental_score(ticker)
    news  = get_news_summary(ticker, name)
    result = get_gemini_signal(ticker, name, tech, fund, news)

    row = {
        "ticker":       ticker,
        "stock_name":   name,
        "signal":       result.get("signal", "HOLD"),
        "confidence":   result.get("confidence", "低"),
        "reason":       result.get("reason", ""),
        "risk":         result.get("risk", ""),
        "price":        tech.get("price"),
        "tech_score":   tech.get("score"),
        "fund_score":   fund.get("score"),
        "tech_detail":  tech.get("detail"),
        "fund_detail":  fund.get("detail"),
        "news_summary": news[:1000],
    }
    db_save_signal(row)
    return row

# ── APScheduler：每日 14:35 自動跑（台股收盤後）─────────────
_scheduler_started = False

def _scheduled_job():
    stocks = db_get_watchlist()
    for s in stocks:
        try:
            run_analysis(s["ticker"], s["name"])
        except Exception as e:
            print(f"排程分析失敗 {s['ticker']}: {e}")

def start_scheduler():
    global _scheduler_started
    if _scheduler_started:
        return
    scheduler = BackgroundScheduler(timezone="Asia/Taipei")
    scheduler.add_job(_scheduled_job, "cron", hour=14, minute=35)
    scheduler.start()
    _scheduler_started = True

# ── Streamlit UI ─────────────────────────────────────────────
st.set_page_config(
    page_title="台股訊號系統",
    page_icon="📈",
    layout="wide",
)

start_scheduler()  # 背景排程啟動

# ── Sidebar：追蹤清單管理 ────────────────────────────────────
with st.sidebar:
    st.title("📋 追蹤清單")

    watchlist = db_get_watchlist()
    for stock in watchlist:
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{stock['name']}** `{stock['ticker']}`")
        if col2.button("移除", key=f"del_{stock['ticker']}"):
            db_remove_stock(stock["ticker"])
            st.rerun()

    st.divider()
    st.subheader("新增股票")
    new_ticker = st.text_input("代碼（含 .TW）", placeholder="例：2412.TW")
    new_name   = st.text_input("名稱", placeholder="例：中華電")
    if st.button("➕ 新增", use_container_width=True):
        if new_ticker and new_name:
            db_add_stock(new_ticker.upper().strip(), new_name.strip())
            st.success(f"已新增 {new_name}")
            st.rerun()
        else:
            st.warning("請填寫代碼與名稱")

    st.divider()
    st.caption("⏰ 每日 14:35 自動分析")
    st.caption("🔑 手動分析在主頁面觸發")

# ── 主頁面 ───────────────────────────────────────────────────
st.title("📈 台股買賣訊號系統")
st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

tab1, tab2, tab3 = st.tabs(["🎯 今日訊號", "🔍 手動分析", "📜 歷史紀錄"])

# ── Tab 1：今日訊號 ──────────────────────────────────────────
with tab1:
    latest = db_get_latest_signals()

    if not latest:
        st.info("尚無訊號資料，請至「手動分析」頁面觸發，或等待每日自動分析（14:35）。")
    else:
        buy  = [s for s in latest if s["signal"] == "BUY"]
        hold = [s for s in latest if s["signal"] == "HOLD"]
        sell = [s for s in latest if s["signal"] == "SELL"]

        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 買進訊號", len(buy))
        col2.metric("🟡 觀望", len(hold))
        col3.metric("🔴 賣出訊號", len(sell))

        st.divider()

        def signal_card(s):
            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(s["signal"], "⚪")
            conf_color = {"高": "🔥", "中": "✳️", "低": "❄️"}.get(s.get("confidence",""), "")
            analyzed = s.get("analyzed_at","")[:16].replace("T"," ")

            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.markdown(f"### {emoji} {s['stock_name']} `{s['ticker']}`")
                c2.markdown(f"**信心度** {conf_color} {s.get('confidence','')}")
                c3.markdown(f"**現價** {s.get('price','–')} 元")

                st.markdown(f"**判斷理由：** {s.get('reason','')}")
                st.caption(f"⚠️ 風險：{s.get('risk','')}　｜　分析時間：{analyzed}")

                with st.expander("詳細指標"):
                    d1, d2 = st.columns(2)
                    d1.markdown(f"**技術面（評分 {s.get('tech_score',0)}）**\n\n{s.get('tech_detail','')}")
                    d2.markdown(f"**基本面（評分 {s.get('fund_score',0)}）**\n\n{s.get('fund_detail','')}")
                    st.markdown("**相關新聞**")
                    st.text(s.get("news_summary","")[:600])

        if buy:
            st.subheader("🟢 買進訊號")
            for s in buy:
                signal_card(s)

        if sell:
            st.subheader("🔴 賣出訊號")
            for s in sell:
                signal_card(s)

        if hold:
            st.subheader("🟡 觀望")
            for s in hold:
                signal_card(s)

# ── Tab 2：手動分析 ──────────────────────────────────────────
with tab2:
    st.subheader("手動觸發分析")
    watchlist = db_get_watchlist()

    if not watchlist:
        st.warning("請先在左側新增追蹤股票。")
    else:
        options = {f"{s['name']} ({s['ticker']})": s for s in watchlist}
        selected_label = st.selectbox("選擇股票", list(options.keys()))
        selected = options[selected_label]

        analyze_all = st.checkbox("分析全部追蹤股票", value=False)

        if st.button("🚀 開始分析", use_container_width=True, type="primary"):
            targets = watchlist if analyze_all else [selected]

            for stock in targets:
                with st.status(f"分析中：{stock['name']} ({stock['ticker']})…", expanded=True) as status:
                    st.write("📊 抓取技術指標...")
                    tech = get_technical_score(stock["ticker"])
                    st.write(f"現價 {tech.get('price')} 元 ｜ 技術評分 {tech['score']}")

                    st.write("📋 抓取基本面...")
                    fund = get_fundamental_score(stock["ticker"])
                    st.write(f"基本面評分 {fund['score']}")

                    st.write("📰 抓取市場新聞...")
                    news = get_news_summary(stock["ticker"], stock["name"])

                    st.write("🤖 Gemini 綜合判斷中...")
                    result = get_gemini_signal(stock["ticker"], stock["name"], tech, fund, news)

                    row = {
                        "ticker":       stock["ticker"],
                        "stock_name":   stock["name"],
                        "signal":       result.get("signal", "HOLD"),
                        "confidence":   result.get("confidence", "低"),
                        "reason":       result.get("reason", ""),
                        "risk":         result.get("risk", ""),
                        "price":        tech.get("price"),
                        "tech_score":   tech.get("score"),
                        "fund_score":   fund.get("score"),
                        "tech_detail":  tech.get("detail"),
                        "fund_detail":  fund.get("detail"),
                        "news_summary": news[:1000],
                    }
                    db_save_signal(row)

                    sig_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(result["signal"], "⚪")
                    status.update(
                        label=f"{sig_emoji} {stock['name']} → {result['signal']} （{result.get('confidence','')}）",
                        state="complete"
                    )

            st.success("✅ 分析完成，切換到「今日訊號」查看結果。")
            time.sleep(1)
            st.rerun()

# ── Tab 3：歷史紀錄 ──────────────────────────────────────────
with tab3:
    st.subheader("歷史訊號紀錄")

    days = st.slider("顯示最近幾天", 1, 90, 30)
    history = db_get_signals(days=days)

    if not history:
        st.info("該時間範圍內無紀錄。")
    else:
        df = pd.DataFrame(history)
        df["analyzed_at"] = pd.to_datetime(df["analyzed_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df["訊號"] = df["signal"].map({"BUY": "🟢 BUY", "HOLD": "🟡 HOLD", "SELL": "🔴 SELL"})

        # 統計
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("總分析次數", len(df))
        col2.metric("買進訊號", len(df[df["signal"] == "BUY"]))
        col3.metric("賣出訊號", len(df[df["signal"] == "SELL"]))
        col4.metric("追蹤股票數", df["ticker"].nunique())

        st.divider()

        # 篩選
        tickers = ["全部"] + sorted(df["ticker"].unique().tolist())
        filter_ticker = st.selectbox("篩選股票", tickers)
        if filter_ticker != "全部":
            df = df[df["ticker"] == filter_ticker]

        display_cols = ["analyzed_at", "stock_name", "ticker", "訊號", "confidence", "price", "reason"]
        st.dataframe(
            df[display_cols].rename(columns={
                "analyzed_at": "分析時間",
                "stock_name":  "股票名稱",
                "ticker":      "代碼",
                "confidence":  "信心度",
                "price":       "價格",
                "reason":      "理由",
            }),
            use_container_width=True,
            hide_index=True,
        )
