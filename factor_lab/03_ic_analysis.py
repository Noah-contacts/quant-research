# ============================================================
#  03_ic_analysis.py — 因子有效性检验（IC分析）
#
#  这个文件做一件事：
#  读取 data/factors/ → 计算每个因子的IC值
#  → 打印报告：哪些因子有效，哪些因子没用
#
#  IC值 = 因子值 与 未来收益率 的相关系数
#  IC均值 > 0.03  → 因子有效，保留
#  IC均值 < 0.03  → 因子无效，剔除或优化
# ============================================================

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from config import DATA_DIR, STOCK_POOL

FACTOR_DIR = os.path.join(DATA_DIR, "factors")

# 要检验的因子列

# 因子中文名（方便看报告）
FACTORS = [
    "UTR_ST",
    "JDQS",
    "LWS",
    "UBL",
    "Neutral_MF",
    "PVI_Refined",
    "APBR",
    "MULTI_SCORE",
     "CTRL_ALPHA",   # timing因子：进入IC分析，但不参与选股评分
]
FACTOR_CN = {
    "UTR_ST" : "STR+Turn20耦合",
    "JDQS"       : "独立强势",
    "LWS"        : "下影线均值",
    "UBL"        : "上影线标准差",
    "Neutral_MF" : "中性化资金流",
    "PVI_Refined": "散户放量跟风",
    "APBR"       : "人气背离",
    "MULTI_SCORE" : "多因子合成",
    "CTRL_ALPHA"  : "主升浪时机因子"
}

def load_all_factors():
    """
    把所有股票的因子CSV合并成一张大表
    这张大表的每一行 = 某只股票在某天的所有因子值
    """
    frames = []
    for code in STOCK_POOL:
        path = os.path.join(FACTOR_DIR, f"{code}_factors.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["ts_code"] = code
        frames.append(df)

    if not frames:
        print("❌ 没有找到因子文件，请先运行 02_factor_calc.py")
        return None

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    return panel


def calc_ic(panel, factor):
    """
    计算单个因子的逐日IC值

    每天的IC = 当天所有股票的因子值 与 未来收益率 的Spearman相关系数

    为什么用Spearman而不是Pearson？
    因为我们只关心排名关系（哪只股票因子值更高），
    不关心具体数值的大小，Spearman更稳健
    """
    ic_list = []

    for date, group in panel.groupby("trade_date"):
        # 取出当天所有股票的因子值和未来收益率
        data = group[[factor, "FUTURE_RET"]].dropna()

        # 至少需要5只股票才能算相关系数
        if len(data) < 5:
            continue

        # 计算Spearman相关系数
        ic, _ = stats.spearmanr(data[factor], data["FUTURE_RET"])
        ic_list.append({"trade_date": date, "IC": ic})

    if not ic_list:
        return None

    return pd.DataFrame(ic_list).set_index("trade_date")


def analyze_factor(ic_series, factor):
    """
    根据IC时间序列，计算因子的综合评分
    """
    ic_values = ic_series["IC"].dropna()

    ic_mean = ic_values.mean()      # IC均值：越大越好
    ic_std  = ic_values.std()       # IC标准差：越小越稳定
    ic_ir   = ic_mean / (ic_std + 1e-9)  # IC_IR：综合衡量因子质量
    ic_pos  = (ic_values > 0).mean() * 100  # IC为正的比例

    # 判断是否有效
    if abs(ic_mean) >= 0.03 and abs(ic_ir) >= 0.3:
        verdict = "✅ 有效"
    elif abs(ic_mean) >= 0.02:
        verdict = "⚠️  较弱"
    else:
        verdict = "❌ 无效"

    return {
        "因子"      : FACTOR_CN.get(factor, factor),
        "IC均值"    : round(ic_mean, 4),
        "IC标准差"  : round(ic_std,  4),
        "IC_IR"     : round(ic_ir,   3),
        "IC正比例"  : f"{ic_pos:.1f}%",
        "结论"      : verdict,
    }


# ── 主程序 ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  因子有效性检验（IC分析）启动")
    print("=" * 60)

    # 加载所有因子数据
    panel = load_all_factors()
    if panel is None:
        exit()

    print(f"  数据加载完成：{len(panel)} 行，{panel['ts_code'].nunique()} 只股票\n")

    # 逐个因子计算IC
    results = []
    valid_factors = []

    for factor in FACTORS:
        if factor not in panel.columns:
            continue

        ic_series = calc_ic(panel, factor)
        if ic_series is None:
            continue

        result = analyze_factor(ic_series, factor)
        results.append(result)

        status = result["结论"]
        if "有效" in status:
            valid_factors.append(factor)

    # 打印结果表格
    print(f"  {'因子':<12} {'IC均值':>8} {'IC_IR':>8} {'IC正比例':>10} {'结论'}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['因子']:<12} {r['IC均值']:>8.4f} {r['IC_IR']:>8.3f} {r['IC正比例']:>10} {r['结论']}")

    # 总结
    print("\n" + "=" * 60)
    print(f"  ✅ 有效因子（{len(valid_factors)}个）：")
    for f in valid_factors:
        print(f"     → {FACTOR_CN.get(f, f)}（{f}）")

    invalid = [f for f in FACTORS if f not in valid_factors]
    print(f"\n  ❌ 待优化因子（{len(invalid)}个）：")
    for f in invalid:
        print(f"     → {FACTOR_CN.get(f, f)}（{f}）")
    print("=" * 60)
    print("\n  下一步：把有效因子的名称记下来，")
    print("  在 04_layer_backtest.py 中进一步验证它们的分层收益。")