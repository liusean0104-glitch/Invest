"""
dcf_engine.py — 無槓桿自由現金流(FCFF)折現估值引擎

設計對應的觀念鏈:
    營收預測 → EBIT → FCFF bridge → WACC 折現 → 終值(TV)
    → 企業價值(EV) → EV-淨負債 bridge → 每股價值 → 敏感度分析

兩種終值算法:Gordon Growth 與 Exit Multiple(EV/EBITDA)。
支援期中折現慣例(mid-year convention)。
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────
# 工具:把單一數字或 list 統一成「每年一個值」的序列
# ──────────────────────────────────────────────────────────────
def _as_series(x: Union[float, list], n: int, name: str) -> list:
    if isinstance(x, (int, float)):
        return [float(x)] * n
    if len(x) != n:
        raise ValueError(f"{name} 的長度 ({len(x)}) 必須等於預測年數 ({n})")
    return [float(v) for v in x]


# ──────────────────────────────────────────────────────────────
# WACC:用 CAPM 算 ke,按市值權重加總
# ──────────────────────────────────────────────────────────────
@dataclass
class WACCInputs:
    risk_free_rate: float          # rf,無風險利率(常用 10 年期公債殖利率)
    beta: float                    # 槓桿後 β(levered beta)
    equity_risk_premium: float     # ERP,市場風險溢酬
    cost_of_debt_pretax: float     # kd,稅前債務成本
    tax_rate: float                # 邊際稅率
    market_cap: float              # E,股權市值
    total_debt: float              # D,總負債(市值,沒有就用帳面近似)

    def cost_of_equity(self) -> float:
        """CAPM: ke = rf + β × ERP"""
        return self.risk_free_rate + self.beta * self.equity_risk_premium

    def cost_of_debt_aftertax(self) -> float:
        """稅後債務成本 = kd × (1 - 稅率)"""
        return self.cost_of_debt_pretax * (1 - self.tax_rate)

    def wacc(self) -> float:
        E, D = self.market_cap, self.total_debt
        V = E + D
        return (E / V) * self.cost_of_equity() + (D / V) * self.cost_of_debt_aftertax()


# ──────────────────────────────────────────────────────────────
# DCF 假設
# ──────────────────────────────────────────────────────────────
@dataclass
class DCFAssumptions:
    base_revenue: float                         # 最近一年(year 0)營收
    revenue_growth: Union[float, list]          # 每年營收成長率
    ebit_margin: Union[float, list]             # 每年 EBIT margin
    tax_rate: float                             # 稅率
    da_pct_revenue: Union[float, list]          # D&A 佔營收 %
    capex_pct_revenue: Union[float, list]       # CapEx 佔營收 %
    nwc_pct_revenue: Union[float, list]         # 營運資金「水位」佔營收 %(用來算 ΔNWC)

    wacc: float                                 # 折現率
    terminal_growth: float                      # 永續成長率 g

    net_debt: float = 0.0                        # 淨負債 = 總負債 - 現金(+少數股權+特別股)
    shares_outstanding: float = 1.0              # 流通股數

    forecast_years: int = 5                      # 明確預測期長度
    exit_ev_ebitda: Optional[float] = None       # 若用 exit multiple 法的終年 EV/EBITDA
    mid_year_convention: bool = False            # 期中折現慣例

    # 終值常態化(只影響 Gordon 法)。穩態下淨資本支出只該支撐成長 g,
    # 若沿用擴張期的 CapEx%,等於把高資本密集度永遠帶進終值,會系統性低估 TV。
    normalize_terminal: bool = True              # True → 終年令 CapEx ≈ D&A
    terminal_ebit_margin: Optional[float] = None     # 終年 margin,None 則沿用最後一年
    terminal_da_pct_revenue: Optional[float] = None  # 終年 D&A%,None 則沿用最後一年
    terminal_nwc_pct_revenue: Optional[float] = None # 終年 NWC%,None 則沿用最後一年


# ──────────────────────────────────────────────────────────────
# 主引擎
# ──────────────────────────────────────────────────────────────
class DCFModel:
    def __init__(self, a: DCFAssumptions):
        self.a = a
        self.n = a.forecast_years
        self._build_projection()

    def _build_projection(self):
        a, n = self.a, self.n
        g       = _as_series(a.revenue_growth,    n, "revenue_growth")
        margin  = _as_series(a.ebit_margin,       n, "ebit_margin")
        da_pct  = _as_series(a.da_pct_revenue,    n, "da_pct_revenue")
        cx_pct  = _as_series(a.capex_pct_revenue, n, "capex_pct_revenue")
        nwc_pct = _as_series(a.nwc_pct_revenue,   n, "nwc_pct_revenue")

        rows = []
        rev_prev = a.base_revenue
        # year 0 的 NWC 水位,用來算第 1 年的 ΔNWC
        # 假設 year 0 用第一年的 NWC% 近似 base NWC(實務上可另外給)
        nwc_prev = a.base_revenue * nwc_pct[0]

        for t in range(1, n + 1):
            i = t - 1
            revenue = rev_prev * (1 + g[i])
            ebit    = revenue * margin[i]
            nopat   = ebit * (1 - a.tax_rate)            # EBIT×(1-稅率)
            da      = revenue * da_pct[i]                # 折舊攤銷(非現金,加回)
            capex   = revenue * cx_pct[i]                # 資本支出
            nwc     = revenue * nwc_pct[i]               # 本年 NWC 水位
            d_nwc   = nwc - nwc_prev                      # ΔNWC(增加 → 占用現金)

            # FCFF = NOPAT + D&A - CapEx - ΔNWC
            fcff = nopat + da - capex - d_nwc

            # 折現期數:期中慣例則為 t-0.5
            period = (t - 0.5) if a.mid_year_convention else t
            df = 1 / (1 + a.wacc) ** period
            pv_fcff = fcff * df

            ebitda = ebit + da  # 給 exit multiple 用

            rows.append({
                "Year": t, "Revenue": revenue, "EBIT": ebit, "EBITDA": ebitda,
                "NOPAT": nopat, "D&A": da, "CapEx": capex, "ΔNWC": d_nwc,
                "FCFF": fcff, "DiscFactor": df, "PV_FCFF": pv_fcff,
            })
            rev_prev, nwc_prev = revenue, nwc

        self.proj = pd.DataFrame(rows).set_index("Year")
        # 存最後一年的各項比率,供終值常態化使用
        self._margin_last  = margin[-1]
        self._da_pct_last  = da_pct[-1]
        self._cx_pct_last  = cx_pct[-1]
        self._nwc_pct_last = nwc_pct[-1]

    # ── 終值 ──────────────────────────────────────────────
    def _terminal_fcff(self) -> float:
        """永續期首年(n+1)的常態化 FCFF,供 Gordon 法使用。

        穩態假設:CapEx 趨近 D&A(淨資本支出只支撐成長),
        NWC 隨營收等比成長 → ΔNWC = 終年水位 − 前一年水位。
        """
        a = self.a
        rev_n = self.proj.iloc[-1]["Revenue"]
        rev_t = rev_n * (1 + a.terminal_growth)          # 永續首年營收

        margin  = a.terminal_ebit_margin    if a.terminal_ebit_margin    is not None else self._margin_last
        da_pct  = a.terminal_da_pct_revenue if a.terminal_da_pct_revenue is not None else self._da_pct_last
        nwc_pct = a.terminal_nwc_pct_revenue if a.terminal_nwc_pct_revenue is not None else self._nwc_pct_last

        ebit  = rev_t * margin
        nopat = ebit * (1 - a.tax_rate)
        da    = rev_t * da_pct
        capex = da if a.normalize_terminal else rev_t * self._cx_pct_last  # 穩態:CapEx ≈ D&A
        d_nwc = rev_t * nwc_pct - rev_n * self._nwc_pct_last
        return nopat + da - capex - d_nwc

    def terminal_value(self, method: str = "gordon"):
        """回傳 (終年未折現 TV, TV 的現值)"""
        a, n = self.a, self.n
        last = self.proj.iloc[-1]

        if method == "gordon":
            if a.wacc <= a.terminal_growth:
                raise ValueError("WACC 必須大於永續成長率 g,否則 Gordon 公式發散")
            # 用常態化的永續首年 FCFF:TV_n = FCFF_{n+1} / (WACC − g)
            tv = self._terminal_fcff() / (a.wacc - a.terminal_growth)
            # 永續流視為期中現金流 → 期中慣例折現期數為 n−0.5
            period = (n - 0.5) if a.mid_year_convention else n
        elif method == "exit_multiple":
            if a.exit_ev_ebitda is None:
                raise ValueError("使用 exit_multiple 法需提供 exit_ev_ebitda")
            tv = last["EBITDA"] * a.exit_ev_ebitda
            # exit multiple 是「第 n 年底賣出」的時點價值 → 一律用整數 n 折現
            period = n
        else:
            raise ValueError("method 只能是 'gordon' 或 'exit_multiple'")

        pv_tv = tv / (1 + a.wacc) ** period
        return tv, pv_tv

    # ── 估值彙總 ────────────────────────────────────────────
    def value(self, method: str = "gordon") -> dict:
        a = self.a
        pv_fcff_sum = self.proj["PV_FCFF"].sum()
        tv, pv_tv = self.terminal_value(method)

        ev = pv_fcff_sum + pv_tv                 # 企業價值
        equity_value = ev - a.net_debt           # EV - 淨負債 = 權益價值
        per_share = equity_value / a.shares_outstanding

        return {
            "method": method,
            "PV_of_FCFF": pv_fcff_sum,
            "TV_undiscounted": tv,
            "PV_of_TV": pv_tv,
            "TV_pct_of_EV": pv_tv / ev,          # 終值佔比(>80% 要警覺)
            "Enterprise_Value": ev,
            "Net_Debt": a.net_debt,
            "Equity_Value": equity_value,
            "Value_per_Share": per_share,
        }

    # ── 敏感度:WACC × 永續成長率 → 每股價值 ──────────────
    def sensitivity(self, wacc_range, g_range, method: str = "gordon") -> pd.DataFrame:
        orig_wacc, orig_g = self.a.wacc, self.a.terminal_growth
        table = pd.DataFrame(index=[f"{w:.1%}" for w in wacc_range],
                             columns=[f"{g:.1%}" for g in g_range], dtype=float)
        for w in wacc_range:
            for g in g_range:
                self.a.wacc, self.a.terminal_growth = w, g
                self._build_projection()  # WACC 變了,折現因子要重算
                try:
                    table.loc[f"{w:.1%}", f"{g:.1%}"] = self.value(method)["Value_per_Share"]
                except ValueError:
                    table.loc[f"{w:.1%}", f"{g:.1%}"] = np.nan
        self.a.wacc, self.a.terminal_growth = orig_wacc, orig_g
        self._build_projection()
        table.index.name = "WACC \\ g"
        return table


# ──────────────────────────────────────────────────────────────
# 示範:一家虛構公司(數字僅供跑通用)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    # 1) 先用 CAPM 算 WACC
    wacc_in = WACCInputs(
        risk_free_rate=0.04, beta=1.10, equity_risk_premium=0.055,
        cost_of_debt_pretax=0.05, tax_rate=0.20,
        market_cap=8000, total_debt=2000,
    )
    print("=" * 60)
    print(f"Cost of Equity (CAPM): {wacc_in.cost_of_equity():.2%}")
    print(f"After-tax Cost of Debt: {wacc_in.cost_of_debt_aftertax():.2%}")
    print(f"WACC: {wacc_in.wacc():.2%}")

    # 2) 設定 DCF 假設
    a = DCFAssumptions(
        base_revenue=5000,
        revenue_growth=[0.12, 0.10, 0.08, 0.06, 0.05],
        ebit_margin=0.22,
        tax_rate=0.20,
        da_pct_revenue=0.05,
        capex_pct_revenue=0.06,
        nwc_pct_revenue=0.10,
        wacc=round(wacc_in.wacc(), 4),
        terminal_growth=0.025,
        net_debt=2000 - 500,        # 總負債 2000 - 現金 500
        shares_outstanding=1000,
        forecast_years=5,
        exit_ev_ebitda=12.0,
        mid_year_convention=False,
    )
    model = DCFModel(a)

    print("\n【現金流預測】")
    print(model.proj[["Revenue", "EBIT", "NOPAT", "D&A", "CapEx", "ΔNWC",
                      "FCFF", "DiscFactor", "PV_FCFF"]].round(1))

    print("\n【估值結果 — Gordon Growth】")
    for k, v in model.value("gordon").items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:,.2f}" + (f"  ({v:.1%})" if "pct" in k else ""))
        else:
            print(f"  {k:<20}: {v}")

    print("\n【估值結果 — Exit Multiple】")
    for k, v in model.value("exit_multiple").items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:,.2f}" + (f"  ({v:.1%})" if "pct" in k else ""))
        else:
            print(f"  {k:<20}: {v}")

    print("\n【敏感度分析:每股價值(WACC × g)】")
    wacc_grid = np.arange(a.wacc - 0.01, a.wacc + 0.011, 0.005)
    g_grid = np.arange(a.terminal_growth - 0.01, a.terminal_growth + 0.011, 0.005)
    print(model.sensitivity(wacc_grid, g_grid, "gordon").round(2))
