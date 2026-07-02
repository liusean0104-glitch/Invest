import os
import math
import pandas as pd
import ta
import yfinance as yf
from supabase import create_client
from dotenv import load_dotenv

# ==========================================
# 1. Supabase 連線
# ==========================================
load_dotenv()
MY_SUPABASE_URL = os.getenv("SUPABASE_URL")
MY_SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(MY_SUPABASE_URL, MY_SUPABASE_KEY)

stock_list = ["2330", "2317", "2454"]

# ==========================================
# 工具
# ==========================================
def safe(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return default
    return v

# ==========================================
# 2. 大盤濾網
# ==========================================
def get_taiex_filter():
    try:
        taiex = yf.download("^TWII", period="2y", auto_adjust=True, progress=False)
        if taiex.empty:
            return False, None, None
        if isinstance(taiex.columns, pd.MultiIndex):
            taiex.columns = taiex.columns.get_level_values(0)
        taiex.columns = [c.lower() for c in taiex.columns]
        taiex["MA200"] = taiex["close"].rolling(200).mean()
        taiex.dropna(subset=["MA200"], inplace=True)
        taiex.reset_index(drop=True, inplace=True)
        if len(taiex) < 3:
            return False, None, None
        last3   = taiex.tail(3)
        is_bear = all(last3["close"].values < last3["MA200"].values)
        return is_bear, round(float(taiex["close"].iloc[-1]), 2), round(float(taiex["MA200"].iloc[-1]), 2)
    except Exception as e:
        print(f"  ⚠ 大盤資料抓取失敗：{e}")
        return False, None, None

# ==========================================
# 3. 指標計算
# ==========================================
def compute_indicators(df, ma_fast=15, ma_slow=60):
    df["MA_FAST"]   = ta.trend.sma_indicator(df["close"], window=ma_fast)
    df["MA_SLOW"]   = ta.trend.sma_indicator(df["close"], window=ma_slow)
    df["RSI_9"]     = ta.momentum.rsi(df["close"], window=9)
    macd_obj        = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["MACD_hist"] = macd_obj.macd_diff()
    bb              = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_upper"]  = bb.bollinger_hband()
    df["BB_mid"]    = bb.bollinger_mavg()
    df["BB_lower"]  = bb.bollinger_lband()
    df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["VOL_MA20"]  = df["volume"].rolling(window=20).mean()
    df.dropna(subset=["MA_FAST", "MA_SLOW", "RSI_9", "MACD_hist",
                       "BB_upper", "BB_mid", "BB_lower", "BB_width", "VOL_MA20"],
              inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ==========================================
# 4. MA 方向
# ==========================================
def ma_direction(df, i):
    if i == 0:
        return 0
    prev = df["MA_FAST"].iloc[i-1] - df["MA_SLOW"].iloc[i-1]
    curr = df["MA_FAST"].iloc[i]   - df["MA_SLOW"].iloc[i]
    if prev <= 0 and curr > 0:  return 1
    if prev >= 0 and curr < 0:  return -1
    return 0

# ==========================================
# 5. 個股崩盤逃命（只在持倉中判斷）
# ==========================================
def panic_sell_triggered(df, i):
    close    = float(df["close"].iloc[i])
    bb_lower = float(df["BB_lower"].iloc[i])
    vol      = float(df["volume"].iloc[i])
    vol_ma20 = float(df["VOL_MA20"].iloc[i])
    # 跌破下軌超過 2% 才算真正恐慌性殺盤，避免「勉強跌破」的假訊號
    return close < bb_lower * 0.98 and vol > vol_ma20 * 2.0

# ==========================================
# 6. 三指標品質評分
# ==========================================
def rsi_quality(rsi_val, direction):
    if direction == 1:
        if rsi_val > 78:        return -1
        if 55 <= rsi_val <= 65: return 1
        return 0
    else:
        if rsi_val < 22:        return -1
        if 35 <= rsi_val <= 45: return 1
        return 0

def macd_quality(df, i, direction):
    if i < 3:
        return 0
    h = [float(df["MACD_hist"].iloc[i - k]) for k in range(3)]
    increasing = h[0] > h[1] > h[2]
    decreasing = h[0] < h[1] < h[2]
    if direction == 1:
        if increasing: return 1
        if decreasing: return -1
        return 0
    else:
        if decreasing: return 1
        if increasing: return -1
        return 0

def bband_quality(df, i, direction):
    close    = float(df["close"].iloc[i])
    bb_upper = float(df["BB_upper"].iloc[i])
    bb_mid   = float(df["BB_mid"].iloc[i])
    bb_lower = float(df["BB_lower"].iloc[i])
    vol      = float(df["volume"].iloc[i])
    vol_ma20 = float(df["VOL_MA20"].iloc[i])
    squeeze  = False
    if i >= 20:
        recent  = df["BB_width"].iloc[max(0, i-60):i+1]
        squeeze = float(df["BB_width"].iloc[i]) < float(recent.quantile(0.20))
    high_volume = vol > vol_ma20 * 1.5
    if direction == 1:
        if close > bb_upper and high_volume and squeeze: return 2
        if close > bb_upper and high_volume:             return 1
        if close > bb_mid:                               return 1
        if close < bb_lower:                             return -1
        return 0
    else:
        if close < bb_lower and high_volume and squeeze: return 2
        if close < bb_lower and high_volume:             return 1
        if close < bb_mid:                               return 1
        if close > bb_upper:                             return -1
        return 0

def quality_score_at(df, i, direction):
    return (
        rsi_quality(float(df["RSI_9"].iloc[i]), direction)
        + macd_quality(df, i, direction)
        + bband_quality(df, i, direction)
    )

# ==========================================
# 7. 今日訊號
# ==========================================
def integrated_today_signal(df, is_bear):
    n = len(df)
    if n < 4:
        return "HOLD", {}

    # 崩盤逃命：只在「目前應持倉狀態」才判斷
    # 用最近一次 MA 交叉方向判斷是否應該持倉
    last_direction = next(
        (ma_direction(df, j) for j in range(n-1, 0, -1) if ma_direction(df, j) != 0), 0
    )
    should_hold_position = (last_direction == 1)

    if should_hold_position and panic_sell_triggered(df, n - 1):
        detail = {
            "ma_fast":         round(float(df["MA_FAST"].iloc[-1]), 2),
            "ma_slow":         round(float(df["MA_SLOW"].iloc[-1]), 2),
            "rsi_9":           round(float(df["RSI_9"].iloc[-1]), 1),
            "direction_today": -1,
            "quality_score":   -99,
            "bear_mode":       is_bear,
            "threshold":       3 if is_bear else 1,
            "trigger":         "PANIC_SELL",
        }
        return "SELL", detail

    quality_threshold = 3 if is_bear else 1
    direction = ma_direction(df, n - 1)

    detail = {
        "ma_fast":         round(float(df["MA_FAST"].iloc[-1]), 2),
        "ma_slow":         round(float(df["MA_SLOW"].iloc[-1]), 2),
        "rsi_9":           round(float(df["RSI_9"].iloc[-1]), 1),
        "direction_today": direction,
        "quality_score":   None,
        "q_rsi": None, "q_macd": None, "q_bband": None,
        "bear_mode":       is_bear,
        "threshold":       quality_threshold,
    }

    if direction == 0:
        return "HOLD", detail

    i       = n - 1
    q_rsi   = rsi_quality(float(df["RSI_9"].iloc[i]), direction)
    q_macd  = macd_quality(df, i, direction)
    q_bband = bband_quality(df, i, direction)
    qs      = q_rsi + q_macd + q_bband

    detail.update({
        "q_rsi": q_rsi, "q_macd": q_macd, "q_bband": q_bband,
        "quality_score": qs,
        "bb_width":  round(float(df["BB_width"].iloc[i]), 4),
        "vol_ratio": round(float(df["volume"].iloc[i]) / float(df["VOL_MA20"].iloc[i]), 2),
    })

    action = ("BUY" if direction == 1 else "SELL") if qs >= quality_threshold else "HOLD"
    return action, detail

# ==========================================
# 8a. 回測：整合策略（含崩盤逃命，只在持倉中觸發）
# ==========================================
def simulate_integrated(df, quality_threshold=1):
    trades       = []
    in_position  = False
    buy_price    = 0.0
    capital      = 1.0
    peak         = 1.0
    max_drawdown = 0.0
    signal_count = {"BUY": 0, "SELL": 0, "HOLD": 0, "PANIC": 0}

    for i in range(1, len(df) - 1):
        next_open = float(df["open"].iloc[i + 1])
        close_i   = float(df["close"].iloc[i])

        if in_position:
            # 持倉中才判斷崩盤逃命
            if panic_sell_triggered(df, i):
                ret      = (next_open - buy_price) / buy_price
                capital *= (1 + ret)
                trades.append(ret)
                in_position = False
                signal_count["PANIC"] += 1
                # 更新 MDD 後繼續下一根 K 線
                unrealized = capital
                if unrealized > peak: peak = unrealized
                dd = (peak - unrealized) / peak
                if dd > max_drawdown: max_drawdown = dd
                continue

            # 持倉中判斷是否要正常賣出
            d = ma_direction(df, i)
            if d == -1:
                qs = quality_score_at(df, i, d)
                if qs >= quality_threshold:
                    ret      = (next_open - buy_price) / buy_price
                    capital *= (1 + ret)
                    trades.append(ret)
                    in_position = False
                    signal_count["SELL"] += 1
                else:
                    signal_count["HOLD"] += 1
            else:
                signal_count["HOLD"] += 1

        else:
            # 沒有持倉：只判斷是否買進
            d = ma_direction(df, i)
            if d == 1:
                qs = quality_score_at(df, i, d)
                if qs >= quality_threshold:
                    buy_price   = next_open
                    in_position = True
                    signal_count["BUY"] += 1
                else:
                    signal_count["HOLD"] += 1
            else:
                signal_count["HOLD"] += 1

        unrealized = capital * (close_i / buy_price) if in_position else capital
        if unrealized > peak: peak = unrealized
        dd = (peak - unrealized) / peak
        if dd > max_drawdown: max_drawdown = dd

    if in_position:
        ret      = (float(df["close"].iloc[-1]) - buy_price) / buy_price
        capital *= (1 + ret)
        trades.append(ret)

    n  = len(trades)
    tr = safe(sum(trades) * 100)
    wr = safe(len([t for t in trades if t > 0]) / n * 100) if n > 0 else 0.0

    return {
        "total_return": round(tr, 2),
        "win_rate":     round(wr, 2),
        "max_drawdown": round(safe(max_drawdown * 100), 2),
        "num_trades":   n,
        "signal_count": signal_count,
    }

# ==========================================
# 8b. 回測：純 MA 對照（無任何過濾）
# ==========================================
def simulate_pure_ma(df):
    trades       = []
    in_position  = False
    buy_price    = 0.0
    capital      = 1.0
    peak         = 1.0
    max_drawdown = 0.0

    for i in range(1, len(df) - 1):
        next_open = float(df["open"].iloc[i + 1])
        close_i   = float(df["close"].iloc[i])
        d         = ma_direction(df, i)

        if d == 1 and not in_position:
            buy_price   = next_open
            in_position = True
        elif d == -1 and in_position:
            ret      = (next_open - buy_price) / buy_price
            capital *= (1 + ret)
            trades.append(ret)
            in_position = False

        unrealized = capital * (close_i / buy_price) if in_position else capital
        if unrealized > peak: peak = unrealized
        dd = (peak - unrealized) / peak
        if dd > max_drawdown: max_drawdown = dd

    if in_position:
        ret      = (float(df["close"].iloc[-1]) - buy_price) / buy_price
        capital *= (1 + ret)
        trades.append(ret)

    n  = len(trades)
    tr = safe(sum(trades) * 100)
    wr = safe(len([t for t in trades if t > 0]) / n * 100) if n > 0 else 0.0

    return {
        "total_return": round(tr, 2),
        "win_rate":     round(wr, 2),
        "max_drawdown": round(safe(max_drawdown * 100), 2),
        "num_trades":   n,
    }

# ==========================================
# 9. 主程式
# ==========================================
print("正在抓取台灣加權指數（大盤濾網）...")
is_bear, taiex_close, taiex_ma200 = get_taiex_filter()
bear_label = "🔴 熊市模式（門檻 ≥3）" if is_bear else "🟢 正常模式（門檻 ≥1）"
print(f"大盤現況：{bear_label}")
if taiex_close and taiex_ma200:
    gap = round((taiex_close - taiex_ma200) / taiex_ma200 * 100, 2)
    print(f"加權指數 {taiex_close}  MA200 {taiex_ma200}  距離 {gap:+.2f}%")

all_backtest_results = []
all_final_signals    = []

for stock in stock_list:
    print(f"\n======== {stock}.TW ========")

    raw_df = yf.download(f"{stock}.TW", period="3y", auto_adjust=True, progress=False)
    if raw_df.empty:
        print(f"⚠ {stock} 無資料，跳過")
        continue

    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    raw_df.columns = [col.lower() for col in raw_df.columns]
    raw_df.reset_index(drop=True, inplace=True)

    # ── 新版：MA15/60 + 品質過濾 + 崩盤逃命 ──
    df_new   = compute_indicators(raw_df.copy(), ma_fast=15, ma_slow=60)
    threshold = 3 if is_bear else 1
    perf_new  = simulate_integrated(df_new, quality_threshold=threshold)
    today_action, today_detail = integrated_today_signal(df_new, is_bear)

    print(f"  【MA15/60 整合策略{'  ⚠ 熊市模式' if is_bear else ''}】")
    print(f"    歷史報酬:{perf_new['total_return']:7.2f}%  勝率:{perf_new['win_rate']:5.1f}%  "
          f"MDD:{perf_new['max_drawdown']:5.2f}%  交易次數:{perf_new['num_trades']}")
    print(f"    訊號分佈：{perf_new['signal_count']}")
    print(f"    今日 MA15={today_detail.get('ma_fast')}  MA60={today_detail.get('ma_slow')}  "
          f"RSI9={today_detail.get('rsi_9')}  方向={today_detail.get('direction_today')}  "
          f"品質分={today_detail.get('quality_score')}  門檻={today_detail.get('threshold')}  "
          f"→ 【{today_action}】")
    if today_detail.get("trigger") == "PANIC_SELL":
        print(f"    🚨 觸發個股崩盤逃命條款！放量跌破布林下軌")
    elif today_detail.get("q_bband") is not None:
        print(f"    布林寬度={today_detail.get('bb_width')}  量比={today_detail.get('vol_ratio')}x  "
              f"q_RSI={today_detail.get('q_rsi')}  q_MACD={today_detail.get('q_macd')}  "
              f"q_BBAND={today_detail.get('q_bband')}")

    # ── 對照：純 MA20/60 無任何過濾 ──
    df_old   = compute_indicators(raw_df.copy(), ma_fast=20, ma_slow=60)
    perf_old = simulate_pure_ma(df_old)
    print(f"  【對照 純MA20/60 無過濾】  報酬:{perf_old['total_return']:7.2f}%  "
          f"勝率:{perf_old['win_rate']:5.1f}%  MDD:{perf_old['max_drawdown']:5.2f}%  "
          f"次數:{perf_old['num_trades']}")

    # ── 存入 Supabase ──
    ma_slow_val = today_detail.get("ma_slow") or 1
    ma_fast_val = today_detail.get("ma_fast") or 0
    ma_gap = round((ma_fast_val - ma_slow_val) / ma_slow_val * 100, 2) if ma_slow_val else None

    all_backtest_results.append({
        "stock_id":     stock,
        "strategy":     "MA15_60_integrated",
        "type":         "trend_filtered",
        "total_return": perf_new["total_return"],
        "win_rate":     perf_new["win_rate"],
        "max_drawdown": perf_new["max_drawdown"],
        "num_trades":   perf_new["num_trades"],
        "today_vote":   1 if today_action == "BUY" else (-1 if today_action == "SELL" else 0),
    })
    all_backtest_results.append({
        "stock_id":     stock,
        "strategy":     "MA20_60_pure",
        "type":         "trend_only",
        "total_return": perf_old["total_return"],
        "win_rate":     perf_old["win_rate"],
        "max_drawdown": perf_old["max_drawdown"],
        "num_trades":   perf_old["num_trades"],
        "today_vote":   None,
    })
    all_final_signals.append({
        "stock_id":      stock,
        "action":        today_action,
        "total_score":   today_detail.get("quality_score"),
        "votes":         str({
            "RSI":   today_detail.get("q_rsi"),
            "MACD":  today_detail.get("q_macd"),
            "BBAND": today_detail.get("q_bband"),
        }),
        "current_price": round(float(raw_df["close"].iloc[-1]), 2),
        "ma_gap_pct":    safe(ma_gap),
        "rsi_value":     safe(today_detail.get("rsi_9")),
    })

# ==========================================
# 10. 寫入 Supabase
# ==========================================
print("\n正在同步至 Supabase...")
try:
    supabase.table("backtest_results").delete().neq("stock_id", "0").execute()
    supabase.table("backtest_results").insert(all_backtest_results).execute()
    supabase.table("final_signals").delete().neq("stock_id", "0").execute()
    supabase.table("final_signals").insert(all_final_signals).execute()
    print("✓ 雙表寫入成功")
except Exception as e:
    print(f"✗ Supabase 寫入失敗：{e}")
