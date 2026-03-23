# ============================================================
#  04_layer_backtest.py — 分层回测
#
#  第1组 = MULTI_SCORE最高（因子最强）
#  第5组 = MULTI_SCORE最低（因子最弱）
#  验证：第1组收益 > 第5组收益 = 因子有效
# ============================================================

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import DATA_DIR, STOCK_POOL, HOLD_PERIOD, REPORT_DIR

FACTOR_DIR = os.path.join(DATA_DIR, "factors")
FACTOR     = "MULTI_SCORE"  # 要回测的因子列
N_GROUPS   = 5
os.makedirs(REPORT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=" * 55)
    print(f"  分层回测：{FACTOR}")
    print(f"  第1组=因子最高，第{N_GROUPS}组=因子最低")
    print("=" * 55)

    # 读取所有股票因子数据，合并成截面面板
    frames = []
    for code in STOCK_POOL:
        path = os.path.join(FACTOR_DIR, f"{code}_factors.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["ts_code"] = code
        frames.append(df)

    if not frames:
        print("❌ 没有因子数据")
        exit()

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(
        panel["trade_date"].astype(str).str[:10]
    )

    if FACTOR not in panel.columns:
        print(f"❌ 因子 {FACTOR} 不存在")
        exit()

    if "FUTURE_RET" not in panel.columns:
        print("❌ FUTURE_RET 不存在")
        exit()

    panel = panel.dropna(subset=[FACTOR, "FUTURE_RET"])
    print(f"  有效样本：{len(panel)} 行，{panel['ts_code'].nunique()} 只股票")

    # 每天截面分组（第1组=最高，第5组=最低）
    def assign_group(x):
        if len(x) < N_GROUPS:
            return pd.Series(np.nan, index=x.index)
        # ascending=False：最高的排第1组
        return pd.qcut(x.rank(method="first", ascending=False),
                       N_GROUPS,
                       labels=range(1, N_GROUPS+1))

    panel["group"] = panel.groupby("trade_date")[FACTOR].transform(assign_group)
    panel = panel.dropna(subset=["group"])
    panel["group"] = panel["group"].astype(int)

    # 每组每天平均收益
    group_ret = panel.groupby(["trade_date", "group"])["FUTURE_RET"].mean().reset_index()
    group_ret.columns = ["trade_date", "group", "ret"]

    # 各组平均日收益
    mean_ret = group_ret.groupby("group")["ret"].mean() * 100

    print("\n  各组平均收益率：")
    print("  " + "-" * 35)
    for g in range(1, N_GROUPS+1):
        bar = "█" * int(abs(mean_ret[g]) * 500)
        print(f"  第{g}组：{mean_ret[g]:+.4f}%  {bar}")

    spread = mean_ret[1] - mean_ret[N_GROUPS]
    print(f"\n  多空收益差（第1组-第{N_GROUPS}组）：{spread:+.4f}%")
    if spread > 0:
        print("  ✅ 因子方向正确，第1组显著跑赢第5组")
    else:
        print("  ❌ 因子方向有问题，需要检查")

    # 累计收益曲线
    group_pivot = group_ret.pivot(index="trade_date", columns="group", values="ret").fillna(0)
    cum_ret = (1 + group_pivot).cumprod()

    # 画图
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"{FACTOR} 因子分层回测报告\n第1组=因子最高，第{N_GROUPS}组=因子最低",
                 fontsize=14, fontweight="bold")

    # 上图：各组平均收益柱状图
    colors = ["#2ECC71","#82E0AA","#F4D03F","#E59866","#E74C3C"]
    bars = axes[0].bar(range(1, N_GROUPS+1),
                       [mean_ret[g] for g in range(1, N_GROUPS+1)],
                       color=colors, width=0.6, edgecolor="white")
    for bar, val in zip(bars, [mean_ret[g] for g in range(1, N_GROUPS+1)]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f"{val:.3f}%", ha="center", va="bottom", fontsize=11)
    axes[0].set_title("各分组平均收益率（%）\n第1组=因子最高，第5组=因子最低")
    axes[0].set_xlabel("分组")
    axes[0].set_ylabel("平均收益率（%）")
    axes[0].set_xticks(range(1, N_GROUPS+1))

    # 下图：第1组 vs 第5组累计收益对比
    axes[1].plot(cum_ret.index, cum_ret[1],
                 color="#2ECC71", linewidth=1.5, label="第1组（高因子）")
    axes[1].plot(cum_ret.index, cum_ret[N_GROUPS],
                 color="#E74C3C", linewidth=1.5, label=f"第{N_GROUPS}组（低因子）")
    axes[1].axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"第1组 vs 第{N_GROUPS}组 累计收益对比")
    axes[1].set_ylabel("累计收益（初始=1）")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, "layer_backtest.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  图表已保存：{save_path}")
    print("=" * 55)