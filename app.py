"""
台股買賣訊號系統 — Streamlit App
pip install streamlit yfinance google-genai requests supabase apscheduler python-dotenv pandas
streamlit run app.py
"""

import os, json, requests, time
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY   = os.getenv("NEWS_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_ANON_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
supabase      = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Supabase ──────────────────────────────────────────────────
def db_get_watchlist():
    return supabase.table("watchlist").select("*").order("added_at").execute().data or []

def db_add_stock(ticker, name):
    supabase.table("watchlist").upsert({"ticker": ticker, "name": name}).execute()

def db_remove_stock(ticker):
    supabase.table("watchlist").delete().eq("ticker", ticker).execute()

def db_save_signal(data):
    supabase.table("signals").insert(data).execute()

def db_get_signals(days=30):
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    return (supabase.table("signals").select("*")
            .gte("analyzed_at", since)
            .order("analyzed_at", desc=True)
            .execute().data or [])

def db_get_latest_signals():
    seen = {}
    for s in db_get_signals(days=3):
        if s["ticker"] not in seen:
            seen[s["ticker"]] = s
    return list(seen.values())

# ── 技術面 ────────────────────────────────────────────────────
def get_technical_score(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if df.empty or len(df) < 60:
        return {"score": 0, "price": None, "detail": "股價資料不足"}

    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    ma20   = close.rolling(20).mean()
    ma60   = close.rolling(60).mean()

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
        "score": score,
        "price": price,
        "ma20":  round(cur_ma20, 2),
        "ma60":  round(cur_ma60, 2),
        "rsi":   round(cur_rsi, 1),
        "detail": "\n".join(signals),
    }

# ── 基本面（TWSE API）────────────────────────────────────────
def get_fundamental_score(ticker):
    stock_id = ticker.replace(".TW", "").replace(".TWO", "")
    pe, dy, rev_g = None, 0, None
    score, signals = 0, []
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        url = ("https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
               f"?stockNo={stock_id}&response=json")
        rows = requests.get(url, headers=headers, timeout=10).json().get("data", [])
        if rows:
            latest = rows[-1]
            try:   dy = float(latest[1]) / 100
            except: dy = 0
            try:   pe = float(latest[3])
            except: pe = None
    except Exception as e:
        signals.append(f"➖ TWSE 抓取失敗：{e}")

    try:
        url2 = ("https://www.twse.com.tw/rwd/zh/finance/t163sb04"
                f"?stockNo={stock_id}&response=json")
        rows2 = requests.get(url2, headers=headers, timeout=10).json().get("data", [])
        if len(rows2) >= 2:
            eps_now  = float(rows2[-1][1])
            eps_prev = float(rows2[-2][1])
            if eps_prev != 0:
                rev_g = (eps_now - eps_prev) / abs(eps_prev)
    except Exception:
        pass

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

# ── 新聞 ─────────────────────────────────────────────────────
def get_news_summary(ticker, name):
    items = []
    try:
        r = requests.get("https://newsapi.org/v2/everything", timeout=10, params={
            "q":        f"{name} OR {ticker.replace('.TW', '')}",
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
            model="gemini-1.5-flash",
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

# ── Gemini 判斷 ───────────────────────────────────────────────
def get_gemini_signal(ticker, name, tech, fund, news):
    prompt = (
        f"你是專業台股分析師。根據以下資料判斷今日 {name}（{ticker}）的操作建議。\n\n"
        f"## 技術面（評分 {tech['score']}）\n"
        f"現價：{tech.get('price')} 元\n"
        f"{tech['detail']}\n\n"
        f"## 基本面（評分 {fund['score']}）\n"
        f"{fund['detail']}\n\n"
        f"## 市場新聞（近7天）\n"
        f"{news}\n\n"
        f"請輸出以下 JSON，不要有任何 markdown 包裝：\n"
        '{{"signal":"BUY"或"HOLD"或"SELL","confidence":"高"或"中"或"低",'
        '"reason":"繁體中文判斷理由含新聞影響，100字內","risk":"主要風險，50字內"}}'
    )
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = resp.text.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        err_msg = str(e)
        st.warning(f"⚠️ Gemini 錯誤：{err_msg[:400]}")
        return {
            "signal":     "HOLD",
            "confidence": "低",
            "reason":     f"API錯誤：{err_msg[:150]}",
            "risk":       "請檢查 Gemini API Key",
        }

# ── 主分析流程 ────────────────────────────────────────────────
def run_analysis(ticker, name):
    tech   = get_technical_score(ticker)
    fund   = get_fundamental_score(ticker)
    news   = get_news_summary(ticker, name)
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

# ── 排程 ─────────────────────────────────────────────────────
_scheduler_started = False

def start_scheduler():
    global _scheduler_started
    if _scheduler_started:
        return
    def job():
        for s in db_get_watchlist():
            try:
                run_analysis(s["ticker"], s["name"])
            except Exception as e:
                print(f"排程失敗 {s['ticker']}: {e}")
    scheduler = BackgroundScheduler(timezone="Asia/Taipei")
    scheduler.add_job(job, "cron", hour=14, minute=35)
    scheduler.start()
    _scheduler_started = True

# ── Streamlit UI ─────────────────────────────────────────────
st.set_page_config(page_title="台股訊號系統", page_icon="📈", layout="wide")
start_scheduler()

# Sidebar
with st.sidebar:
    st.title("📋 追蹤清單")
    watchlist = db_get_watchlist()
    for stock in watchlist:
        c1, c2 = st.columns([3, 1])
        c1.write(f"**{stock['name']}** `{stock['ticker']}`")
        if c2.button("移除", key=f"del_{stock['ticker']}"):
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

# Main
st.title("📈 台股買賣訊號系統")
st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

tab1, tab2, tab3 = st.tabs(["🎯 今日訊號", "🔍 手動分析", "📜 歷史紀錄"])

# Tab1
with tab1:
    latest = db_get_latest_signals()
    if not latest:
        st.info("尚無訊號，請至「手動分析」頁面觸發，或等待每日自動分析（14:35）。")
    else:
        buy  = [s for s in latest if s["signal"] == "BUY"]
        hold = [s for s in latest if s["signal"] == "HOLD"]
        sell = [s for s in latest if s["signal"] == "SELL"]
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 買進訊號", len(buy))
        c2.metric("🟡 觀望", len(hold))
        c3.metric("🔴 賣出訊號", len(sell))
        st.divider()

        def signal_card(s):
            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(s["signal"], "⚪")
            conf_icon = {"高": "🔥", "中": "✳️", "低": "❄️"}.get(s.get("confidence", ""), "")
            analyzed = s.get("analyzed_at", "")[:16].replace("T", " ")
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.markdown(f"### {emoji} {s['stock_name']} `{s['ticker']}`")
                c2.markdown(f"**信心度** {conf_icon} {s.get('confidence', '')}")
                c3.markdown(f"**現價** {s.get('price', '–')} 元")
                st.markdown(f"**判斷理由：** {s.get('reason', '')}")
                st.caption(f"⚠️ 風險：{s.get('risk', '')}　｜　分析時間：{analyzed}")
                with st.expander("詳細指標"):
                    d1, d2 = st.columns(2)
                    d1.markdown(f"**技術面（評分 {s.get('tech_score', 0)}）**\n\n{s.get('tech_detail', '')}")
                    d2.markdown(f"**基本面（評分 {s.get('fund_score', 0)}）**\n\n{s.get('fund_detail', '')}")
                    st.markdown("**相關新聞**")
                    st.text(s.get("news_summary", "")[:600])

        if buy:
            st.subheader("🟢 買進訊號")
            for s in buy: signal_card(s)
        if sell:
            st.subheader("🔴 賣出訊號")
            for s in sell: signal_card(s)
        if hold:
            st.subheader("🟡 觀望")
            for s in hold: signal_card(s)

# Tab2
with tab2:
    st.subheader("手動觸發分析")
    watchlist = db_get_watchlist()
    if not watchlist:
        st.warning("請先在左側新增追蹤股票。")
    else:
        options = {f"{s['name']} ({s['ticker']})": s for s in watchlist}
        selected = options[st.selectbox("選擇股票", list(options.keys()))]
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

                    db_save_signal({
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
                    })
                    sig_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(result["signal"], "⚪")
                    status.update(
                        label=f"{sig_emoji} {stock['name']} → {result['signal']} （{result.get('confidence', '')}）",
                        state="complete"
                    )
            st.success("✅ 分析完成，切換到「今日訊號」查看結果。")
            time.sleep(1)
            st.rerun()

# Tab3
with tab3:
    st.subheader("歷史訊號紀錄")
    days    = st.slider("顯示最近幾天", 1, 90, 30)
    history = db_get_signals(days=days)
    if not history:
        st.info("該時間範圍內無紀錄。")
    else:
        df = pd.DataFrame(history)
        df["analyzed_at"] = pd.to_datetime(df["analyzed_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df["訊號"] = df["signal"].map({"BUY": "🟢 BUY", "HOLD": "🟡 HOLD", "SELL": "🔴 SELL"})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總分析次數", len(df))
        c2.metric("買進訊號", len(df[df["signal"] == "BUY"]))
        c3.metric("賣出訊號", len(df[df["signal"] == "SELL"]))
        c4.metric("追蹤股票數", df["ticker"].nunique())
        st.divider()
        tickers = ["全部"] + sorted(df["ticker"].unique().tolist())
        ft = st.selectbox("篩選股票", tickers)
        if ft != "全部":
            df = df[df["ticker"] == ft]
        st.dataframe(
            df[["analyzed_at", "stock_name", "ticker", "訊號", "confidence", "price", "reason"]].rename(columns={
                "analyzed_at": "分析時間", "stock_name": "股票名稱", "ticker": "代碼",
                "confidence": "信心度", "price": "價格", "reason": "理由",
            }),
            use_container_width=True,
            hide_index=True,
        )
