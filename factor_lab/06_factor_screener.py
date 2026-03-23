# ============================================================
#  07_screener.py — 实盘选股工具
#
#  输出：每只候选股的
#    - 介入价格（当前收盘价 / 江恩支撑回踩价）
#    - 止损价格（ATR动态止损 + 固定6%兜底取较高者）
#    - 持仓周期（建议持有天数）
#    - 综合评分（MULTI_SCORE + 子因子确认）
#
#  运行：python factor_lab/07_screener.py
# ============================================================

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from config import DATA_DIR, STOCK_POOL

FACTOR_DIR = os.path.join(DATA_DIR, "factors")

# ══════════════════════════════════════════════════════════════
#  参数
# ══════════════════════════════════════════════════════════════
TOP_N          = 15      # 输出前N只
SCORE_ENTRY    = 0.10    # 因子门槛
STOP_LOSS_PCT  = 0.06    # 固定止损比例
ATR_WINDOW     = 14      # ATR窗口
ATR_MULT       = 2.0     # ATR止损倍数
GANN_SWING     = 20      # 江恩回看窗口（交易日）
GANN_SLOPE_DAY = 0.0015  # 江恩百分比日斜率
GANN_TOL       = 0.05    # 江恩支撑容忍带（±5%）
HOLD_DAYS      = 20      # 基础持仓周期（交易日）


# ══════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════

def load_factor_cache():
    cache = {}
    for code in STOCK_POOL:
        path = os.path.join(FACTOR_DIR, f"{code}_factors.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if "MULTI_SCORE" not in df.columns:
                continue
            df["trade_date"] = pd.to_datetime(
                df["trade_date"].astype(str).str[:10]
            )
            df = df.set_index("trade_date").sort_index()
            cache[code] = df
        except Exception:
            continue
    return cache


def load_price(code):
    path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["trade_date"] = pd.to_datetime(
            df["trade_date"].astype(str).str[:10]
        )
        return df.set_index("trade_date").sort_index()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
#  技术指标
# ══════════════════════════════════════════════════════════════

def calc_atr(price_df):
    if price_df is None or len(price_df) < ATR_WINDOW + 1:
        return None
    df = price_df.tail(ATR_WINDOW + 1).copy()
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return float(tr.mean())


def calc_gann(price_df, slope_mult):
    """百分比斜率江恩线"""
    if price_df is None or len(price_df) < GANN_SWING:
        return None
    seg = price_df.tail(GANN_SWING)
    low_price  = float(seg["low"].min())
    low_date   = seg["low"].idxmin()
    last_date  = price_df.index[-1]
    days_since = (last_date - low_date).days
    if days_since <= 0:
        return low_price
    return low_price * (1.0 + GANN_SLOPE_DAY * slope_mult * days_since)


def near_support(price_df, close):
    """判断是否回踩江恩支撑"""
    for mult in [0.5, 1.0]:
        sup = calc_gann(price_df, mult)
        if sup and sup < close <= sup * (1 + GANN_TOL):
            return True, sup, mult
    return False, None, None


# ══════════════════════════════════════════════════════════════
#  截面排名
# ══════════════════════════════════════════════════════════════

def build_cross_section(cache, latest_date):
    """取最新交易日的截面排名"""
    rows = {}
    for code, df in cache.items():
        if latest_date in df.index:
            rows[code] = df.loc[latest_date]
    if len(rows) < 10:
        return {}

    panel = pd.DataFrame(rows).T
    ranks = {}
    for f in ["UTR_ST", "LWS", "PVI_Refined", "APBR"]:
        if f in panel.columns:
            ranks[f] = panel[f].rank(pct=True).to_dict()
    return ranks


# ══════════════════════════════════════════════════════════════
#  选股核心
# ══════════════════════════════════════════════════════════════

def screen(cache):
    # 找最新因子日期
    latest_date = max(df.index[-1] for df in cache.values())
    ranks = build_cross_section(cache, latest_date)

    results = []

    for code, df in cache.items():
        if latest_date not in df.index:
            continue

        row   = df.loc[latest_date]
        score = float(row.get("MULTI_SCORE", np.nan))
        if np.isnan(score) or score < SCORE_ENTRY:
            continue

        # 价格数据
        price_df = load_price(code)
        if price_df is None or len(price_df) < 30:
            continue

        close = float(price_df["close"].iloc[-1])

        # ── 止损价 ────────────────────────────────────────────
        atr = calc_atr(price_df)
        atr_stop  = (close - ATR_MULT * atr) if atr else None
        hard_stop = close * (1 - STOP_LOSS_PCT)
        # 取较高者（更紧的止损）
        if atr_stop:
            stop_price = max(atr_stop, hard_stop)
            stop_type  = "ATR" if atr_stop >= hard_stop else "固定6%"
        else:
            stop_price = hard_stop
            stop_type  = "固定6%"

        stop_pct = (close - stop_price) / close * 100

        # ── 介入价格 & 江恩加分 ───────────────────────────────
        hit, sup, mult = near_support(price_df, close)
        if hit:
            # 回踩支撑：当前价就是介入价，加分
            entry_price = close
            entry_note  = f"回踩江恩{'1:2' if mult==0.5 else '1:1'}支撑 ¥{sup:.2f}"
            final_score = score * 1.2
            entry_type  = "支撑回踩"
        else:
            # 普通候选：介入价 = 当前价
            entry_price = close
            entry_note  = "当前价介入"
            final_score = score
            entry_type  = "普通"

        # ── 子因子确认 ────────────────────────────────────────
        utr_rank = ranks.get("UTR_ST", {}).get(code, np.nan)
        lws_rank = ranks.get("LWS",    {}).get(code, np.nan)
        pvi_rank = ranks.get("PVI_Refined", {}).get(code, np.nan)

        # 筹码稳定：UTR_ST排名靠后（负向因子）
        utr_ok = (not np.isnan(utr_rank)) and utr_rank >=0.60        # 下影线支撑：LWS排名靠前
        lws_ok = (not np.isnan(lws_rank)) and lws_rank >= 0.50
        # 散户未过热：PVI排名靠后
        pvi_ok = (not np.isnan(pvi_rank)) and pvi_rank <= 0.70

        sub_pass = sum([utr_ok, lws_ok, pvi_ok])

        # ── 持仓周期建议 ──────────────────────────────────────
        # 分数越高 + 江恩加分 → 可以持仓更长
        if final_score > 0.8 and hit:
            hold_days = 25
            hold_note = "强信号，建议持满周期"
        elif final_score > 0.5:
            hold_days = 20
            hold_note = "正常周期"
        else:
            hold_days = 15
            hold_note = "信号偏弱，缩短周期"

        # ── 江恩压力位（目标价参考）──────────────────────────
        res_21 = calc_gann(price_df, slope_mult=2.0)
        res_41 = calc_gann(price_df, slope_mult=4.0)

        results.append({
            "code":        code,
            "score":       score,
            "final_score": final_score,
            "entry_price": entry_price,
            "entry_type":  entry_type,
            "entry_note":  entry_note,
            "stop_price":  round(stop_price, 2),
            "stop_pct":    round(stop_pct, 1),
            "stop_type":   stop_type,
            "hold_days":   hold_days,
            "hold_note":   hold_note,
            "res_21":      round(res_21, 2) if res_21 else None,
            "res_41":      round(res_41, 2) if res_41 else None,
            "sub_pass":    sub_pass,
            "utr_ok":      utr_ok,
            "lws_ok":      lws_ok,
            "pvi_ok":      pvi_ok,
            "utr_rank":    utr_rank,
            "lws_rank":    lws_rank,
            "pvi_rank":    pvi_rank,
            "atr":         round(atr, 3) if atr else None,
        })

    # 按final_score排序
    results.sort(key=lambda x: -x["final_score"])
    return results[:TOP_N], latest_date


# ══════════════════════════════════════════════════════════════
#  打印输出
# ══════════════════════════════════════════════════════════════

def print_report(results, latest_date):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print()
    print("=" * 70)
    print(f"  ⚡ 量化选股系统  |  因子日期：{latest_date.date()}  |  {now}")
    print("=" * 70)
    print(f"  筛选结果：共 {len(results)} 只候选股（按综合评分排序）")
    print()

    # 分类输出
    gann_hits = [r for r in results if r["entry_type"] == "支撑回踩"]
    normals   = [r for r in results if r["entry_type"] == "普通"]

    if gann_hits:
        print("  ★ 江恩支撑回踩（优先级最高）")
        print("  " + "─" * 66)
        for r in gann_hits:
            _print_stock(r)

    if normals:
        print("  ◆ 普通因子候选")
        print("  " + "─" * 66)
        for r in normals:
            _print_stock(r)

    print()
    print("=" * 70)
    print("  📌 操作说明")
    print("  " + "─" * 66)
    print("  • 介入价：明日开盘附近挂单，偏离超过1%则放弃")
    print("  • 止损价：买入后立即设置条件单，严格执行不犹豫")
    print("  • 持仓周期：到期当日收盘前卖出，不论盈亏")
    print("  • ★标记：江恩支撑回踩信号更强，优先考虑")
    print("  • 压力位：可作为止盈参考，非强制")
    print("=" * 70)
    print()


def _print_stock(r):
    # 子因子标记
    utr_mark = "✅" if r["utr_ok"] else "❌"
    lws_mark = "✅" if r["lws_ok"] else "❌"
    pvi_mark = "✅" if r["pvi_ok"] else "❌"
    sub_str  = f"筹码{utr_mark} 支撑{lws_mark} 情绪{pvi_mark}"

    # 压力位
    res_str = ""
    if r["res_21"]:
        res_str += f"  压力1(2:1): ¥{r['res_21']}"
    if r["res_41"]:
        res_str += f"  压力2(4:1): ¥{r['res_41']}"

    print(f"""
  [{r['code']}]  评分:{r['final_score']:.4f}  子因子:{sub_str}
  ┌─ 介入价  ¥{r['entry_price']:.2f}   {r['entry_note']}
  ├─ 止损价  ¥{r['stop_price']:.2f}   ({r['stop_type']} -{r['stop_pct']:.1f}%)
  ├─ 持仓期  {r['hold_days']} 交易日   {r['hold_note']}
  └─ 参考压力{res_str if res_str else '  暂无'}""")


# ══════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  加载因子数据...")
    cache = load_factor_cache()
    print(f"  共加载 {len(cache)} 只股票因子")

    print("  计算截面排名 + 选股中...\n")
    results, latest_date = screen(cache)

    print_report(results, latest_date)