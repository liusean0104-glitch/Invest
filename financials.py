"""
financials.py — 財報品質分析

抓 yfinance 三表（損益表 / 資產負債表 / 現金流量表），計算「品質型」指標：
    ROIC、自由現金流(FCF)、FCF 轉換率、毛利率趨勢、淨負債

這些是用來「佐證」訊號的財報證據，不是價格指標。
另外提供 auto_dcf_inputs()，自動從財報帶出 DCF 基期假設。
"""

from functools import lru_cache
import yfinance as yf


# ── 工具 ──────────────────────────────────────────────────────
def _row(df, *names):
    """在財報 DataFrame 裡找指定列名（多個候選名，回傳第一個找到的 Series）。"""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna()
            if not s.empty:
                return s
    return None


def _latest(s, i=0):
    """取 Series 的第 i 新的值（i=0 最新，i=1 前一年）。"""
    if s is None or len(s) <= i:
        return None
    try:
        return float(s.iloc[i])
    except Exception:
        return None


@lru_cache(maxsize=128)
def fetch_financials(ticker):
    """
    抓三表，回傳 dict（含 income / balance / cashflow DataFrame）。
    用 lru_cache 避免同一進程內重複打 API（緩解 rate limit）。
    抓不到回傳 None。
    """
    try:
        t = yf.Ticker(ticker)
        inc = t.income_stmt
        bs  = t.balance_sheet
        cf  = t.cashflow
        if inc is None or inc.empty:
            return None
        return {"income": inc, "balance": bs, "cashflow": cf}
    except Exception:
        return None


# ── 財報品質評分 ──────────────────────────────────────────────
def get_financial_quality(ticker, assumed_wacc=0.10):
    """
    回傳 dict：score（整數）、detail（文字佐證）、metrics（原始數字）。
    評分對應「定價權 + 資本效率 + 現金流真實性」框架。
    """
    fin = fetch_financials(ticker)
    if not fin:
        return {"score": 0, "detail": "財報資料暫時無法取得（可能 API 限流，稍後重試）", "metrics": {}}

    inc, bs, cf = fin["income"], fin["balance"], fin["cashflow"]

    # 損益表
    revenue   = _row(inc, "Total Revenue", "Operating Revenue")
    ebit      = _row(inc, "EBIT", "Operating Income")
    net_inc   = _row(inc, "Net Income", "Net Income Common Stockholders")
    gross     = _row(inc, "Gross Profit")
    pretax    = _row(inc, "Pretax Income", "Pre Tax Income")
    tax       = _row(inc, "Tax Provision", "Income Tax Expense")

    # 資產負債表
    debt      = _row(bs, "Total Debt", "Total Debt And Capital Lease Obligation")
    cash      = _row(bs, "Cash And Cash Equivalents",
                     "Cash Cash Equivalents And Short Term Investments")
    equity    = _row(bs, "Stockholders Equity", "Common Stock Equity",
                     "Total Equity Gross Minority Interest")
    inv_cap   = _row(bs, "Invested Capital")

    # 現金流量表
    ocf       = _row(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex     = _row(cf, "Capital Expenditure", "Purchase Of PPE")

    score, sig = 0, []
    m = {}

    # ── 1) ROIC（資本效率）──────────────────────────────
    rev0   = _latest(revenue)
    ebit0  = _latest(ebit)
    tax_rate = 0.20
    pt, tx = _latest(pretax), _latest(tax)
    if pt and tx and pt != 0:
        tax_rate = max(0.0, min(0.45, tx / pt))     # 夾在合理區間

    ic = _latest(inv_cap)
    if ic is None:
        d, c, e = _latest(debt) or 0, _latest(cash) or 0, _latest(equity) or 0
        ic = d + e - c if (d + e - c) > 0 else None

    if ebit0 and ic and ic > 0:
        nopat = ebit0 * (1 - tax_rate)
        roic  = nopat / ic
        m["roic"] = roic
        if roic > 0.15:
            score += 1; sig.append(f"✅ ROIC {roic*100:.1f}%（資本效率優異，遠高於資金成本）")
        elif roic > assumed_wacc:
            sig.append(f"➖ ROIC {roic*100:.1f}%（高於資金成本，仍在創造價值）")
        else:
            score -= 1; sig.append(f"❌ ROIC {roic*100:.1f}%（低於資金成本，恐在毀滅價值）")
    else:
        sig.append("➖ ROIC 資料不足")

    # ── 2) 自由現金流轉換率（現金流真實性）──────────────
    ocf0, cx0, ni0 = _latest(ocf), _latest(capex), _latest(net_inc)
    if ocf0 is not None and cx0 is not None:
        fcf = ocf0 + cx0                            # capex 在 yfinance 為負值
        m["fcf"] = fcf
        if ni0 and ni0 != 0:
            conv = fcf / ni0
            m["fcf_conversion"] = conv
            if conv > 0.9:
                score += 1; sig.append(f"✅ FCF 轉換率 {conv*100:.0f}%（利潤幾乎全變現金，品質高）")
            elif conv > 0.6:
                sig.append(f"➖ FCF 轉換率 {conv*100:.0f}%（尚可）")
            else:
                score -= 1; sig.append(f"❌ FCF 轉換率 {conv*100:.0f}%（賺到的錢留不住，利潤品質差）")
        if rev0:
            sig.append(f"　自由現金流 {fcf/1e8:.1f} 億，FCF margin {fcf/rev0*100:.1f}%")
    else:
        sig.append("➖ 現金流量資料不足")

    # ── 3) 毛利率趨勢（定價權）──────────────────────────
    g0, g2 = _latest(gross, 0), _latest(gross, 2)
    r0, r2 = _latest(revenue, 0), _latest(revenue, 2)
    if g0 and r0 and g2 and r2 and r0 != 0 and r2 != 0:
        gm_now, gm_old = g0 / r0, g2 / r2
        m["gross_margin"] = gm_now
        diff = (gm_now - gm_old) * 100
        if diff > 1.5:
            score += 1; sig.append(f"✅ 毛利率擴張（{gm_old*100:.1f}% → {gm_now*100:.1f}%，定價權增強）")
        elif diff < -1.5:
            score -= 1; sig.append(f"❌ 毛利率萎縮（{gm_old*100:.1f}% → {gm_now*100:.1f}%，恐在降價競爭）")
        else:
            sig.append(f"➖ 毛利率穩定（約 {gm_now*100:.1f}%）")
    elif g0 and r0:
        sig.append(f"　毛利率 {g0/r0*100:.1f}%（趨勢資料不足）")

    # ── 4) 淨負債（財務體質）──────────────────────────
    d0, c0 = _latest(debt), _latest(cash)
    if d0 is not None and c0 is not None:
        net_debt = d0 - c0
        m["net_debt"] = net_debt
        if net_debt < 0:
            sig.append(f"✅ 淨現金 {-net_debt/1e8:.1f} 億（體質穩健）")
        else:
            sig.append(f"➖ 淨負債 {net_debt/1e8:.1f} 億")

    return {
        "score":   score,
        "detail":  "\n".join(sig) if sig else "財報資料不足",
        "metrics": m,
    }


# ── DCF 基期假設自動帶入 ──────────────────────────────────────
def auto_dcf_inputs(ticker):
    """
    從財報自動推算 DCF 基期假設，回傳 dict（抓不到的欄位為 None）。
    使用者仍可在介面手動覆寫。
    """
    fin = fetch_financials(ticker)
    if not fin:
        return None
    inc, bs, cf = fin["income"], fin["balance"], fin["cashflow"]

    revenue = _row(inc, "Total Revenue", "Operating Revenue")
    ebit    = _row(inc, "EBIT", "Operating Income")
    pretax  = _row(inc, "Pretax Income", "Pre Tax Income")
    tax     = _row(inc, "Tax Provision", "Income Tax Expense")
    da      = _row(cf, "Depreciation And Amortization",
                  "Depreciation Amortization Depletion")
    capex   = _row(cf, "Capital Expenditure", "Purchase Of PPE")
    wc      = _row(bs, "Working Capital")
    debt    = _row(bs, "Total Debt", "Total Debt And Capital Lease Obligation")
    cash    = _row(bs, "Cash And Cash Equivalents",
                   "Cash Cash Equivalents And Short Term Investments")

    rev0 = _latest(revenue)
    if not rev0 or rev0 == 0:
        return None

    out = {"base_revenue": rev0}

    e0 = _latest(ebit)
    out["ebit_margin"] = round(e0 / rev0, 4) if e0 else 0.15

    pt, tx = _latest(pretax), _latest(tax)
    out["tax_rate"] = round(max(0.0, min(0.45, tx / pt)), 4) if (pt and tx and pt != 0) else 0.20

    d0 = _latest(da)
    out["da_pct"] = round(d0 / rev0, 4) if d0 else 0.05

    cx0 = _latest(capex)
    out["capex_pct"] = round(abs(cx0) / rev0, 4) if cx0 else 0.06

    w0 = _latest(wc)
    out["nwc_pct"] = round(abs(w0) / rev0, 4) if w0 else 0.10

    dd, cc = _latest(debt) or 0, _latest(cash) or 0
    out["net_debt"] = dd - cc

    return out
