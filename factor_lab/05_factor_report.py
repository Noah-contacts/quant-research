# ============================================================
#  06_rf_factor_weight.py — 随机森林因子权重优化
#
#  功能：
#  1. 用随机森林回归预测 FUTURE_RET，获取特征重要性
#  2. 用SHAP值分析各因子的方向性（正向/负向）
#  3. 输出最优权重配比，直接可用于 MULTI_SCORE 合成
#  4. 时序分组交叉验证，防止未来数据泄露
#
#  运行：python factor_lab/06_rf_factor_weight.py
# ============================================================

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from scipy import stats

from config import DATA_DIR, STOCK_POOL, REPORT_DIR

FACTOR_DIR = os.path.join(DATA_DIR, "factors")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── 参与优化的因子列表（只选IC分析中有意义的）──────────────────
CANDIDATE_FACTORS = [
    "UTR_ST",
    "LWS",
    "UBL",
    "Neutral_MF",
    "PVI_Refined",
    "APBR",
    "JDQS",
]
TARGET_COL = "FUTURE_RET"


# ══════════════════════════════════════════════════════════════
#  Step 1: 加载所有因子数据，合并为面板
# ══════════════════════════════════════════════════════════════

def load_panel():
    print("  加载因子面板...")
    frames = []
    for code in STOCK_POOL:
        path = os.path.join(FACTOR_DIR, f"{code}_factors.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df["ts_code"] = code
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("没有找到因子文件，请先运行 02_factor_calc.py")

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(
        panel["trade_date"].astype(str).str[:10]
    )
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    print(f"  面板大小：{len(panel)} 行，{panel['ts_code'].nunique()} 只股票")
    print(f"  时间范围：{panel['trade_date'].min().date()} ~ "
          f"{panel['trade_date'].max().date()}")
    return panel


# ══════════════════════════════════════════════════════════════
#  Step 2: 截面Rank标准化（防止量纲差异影响RF）
# ══════════════════════════════════════════════════════════════

def cross_section_rank(panel, factors):
    """
    每个交易日内，对各因子做截面排名（pct rank → zscore）
    这样RF看到的输入和MULTI_SCORE合成时的输入一致
    """
    print("  截面Rank标准化...")
    result = panel.copy()

    for f in factors:
        if f not in panel.columns:
            print(f"  ⚠ 因子 {f} 不存在，跳过")
            continue
        ranked = panel.groupby("trade_date")[f].rank(pct=True)
        # pct rank → zscore
        result[f"RANK_{f}"] = (ranked - 0.5) * 2   # 映射到 [-1, 1]

    return result


# ══════════════════════════════════════════════════════════════
#  Step 3: 时序分组交叉验证（防止未来泄露）
# ══════════════════════════════════════════════════════════════

def time_series_cv_split(dates, n_splits=5):
    """
    时序分组：按时间顺序切分，训练集永远在测试集之前
    避免用未来数据训练模型
    """
    unique_dates = sorted(dates.unique())
    total = len(unique_dates)
    fold_size = total // (n_splits + 1)

    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end  = min(train_end + fold_size, total)
        train_dates = set(unique_dates[:train_end])
        test_dates  = set(unique_dates[train_end:test_end])
        splits.append((train_dates, test_dates))

    return splits


# ══════════════════════════════════════════════════════════════
#  Step 4: 随机森林训练 + 特征重要性
# ══════════════════════════════════════════════════════════════

def train_rf(X, y):
    """训练随机森林，返回模型和特征重要性"""
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,          # 浅树，防止过拟合
        min_samples_leaf=50,  # 叶子节点最少50样本
        max_features=0.6,     # 每次分裂用60%特征
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X, y)
    return rf


def train_gbm(X, y):
    """GBM作为对照，验证RF结果稳健性"""
    gbm = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=50,
        random_state=42
    )
    gbm.fit(X, y)
    return gbm


# ══════════════════════════════════════════════════════════════
#  Step 5: 方向性分析（IC符号）
# ══════════════════════════════════════════════════════════════

def calc_factor_directions(panel, rank_factors, target):
    """
    计算每个因子的截面IC，确定方向（正向/负向）
    RF特征重要性只有大小没有方向，必须配合IC确定符号
    """
    print("\n  计算因子方向（截面IC）...")
    directions = {}
    ic_values  = {}

    for rf in rank_factors:
        f = rf.replace("RANK_", "")
        ics = []
        for date, group in panel.groupby("trade_date"):
            g = group[[rf, target]].dropna()
            if len(g) < 20:
                continue
            ic, _ = stats.spearmanr(g[rf], g[target])
            ics.append(ic)

        if ics:
            mean_ic = np.mean(ics)
            ic_values[f]  = mean_ic
            directions[f] = 1 if mean_ic > 0 else -1
            sign = "正向(+)" if mean_ic > 0 else "负向(-)"
            print(f"    {f:20s}  IC={mean_ic:+.4f}  {sign}")
        else:
            ic_values[f]  = 0
            directions[f] = 1

    return directions, ic_values


# ══════════════════════════════════════════════════════════════
#  Step 6: 计算最优权重
# ══════════════════════════════════════════════════════════════

def calc_optimal_weights(importances, directions, ic_values, threshold=0.03):
    """
    最优权重 = RF特征重要性 × IC方向符号
    过滤掉IC绝对值 < threshold 的无效因子
    """
    weights = {}
    for factor, importance in importances.items():
        ic = ic_values.get(factor, 0)
        if abs(ic) < threshold:
            weights[factor] = 0.0   # IC太低，直接过滤
            continue
        direction = directions.get(factor, 1)
        weights[factor] = importance * direction

    # 归一化：使绝对值之和为1
    total = sum(abs(w) for w in weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


# ══════════════════════════════════════════════════════════════
#  Step 7: 可视化
# ══════════════════════════════════════════════════════════════

def plot_results(importances_rf, importances_gbm, weights,
                 ic_values, cv_scores, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("随机森林因子权重优化报告", fontsize=14, fontweight="bold")

    factors = list(importances_rf.keys())
    colors  = ["#2ecc71" if w >= 0 else "#e74c3c"
               for w in [weights.get(f, 0) for f in factors]]

    # ── 图1：RF特征重要性 ──────────────────────────────────────
    ax = axes[0, 0]
    vals = [importances_rf[f] for f in factors]
    bars = ax.barh(factors, vals, color="steelblue", alpha=0.8)
    ax.set_title("RF 特征重要性（大小）")
    ax.set_xlabel("Importance")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    # ── 图2：GBM特征重要性（对照）────────────────────────────
    ax = axes[0, 1]
    vals_gbm = [importances_gbm.get(f, 0) for f in factors]
    ax.barh(factors, vals_gbm, color="coral", alpha=0.8)
    ax.set_title("GBM 特征重要性（对照验证）")
    ax.set_xlabel("Importance")

    # ── 图3：最优权重（含方向）───────────────────────────────
    ax = axes[1, 0]
    final_weights = [weights.get(f, 0) for f in factors]
    bars = ax.barh(factors, final_weights, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("最优合成权重（含方向）\n正=正向因子，负=负向因子")
    ax.set_xlabel("Weight")
    for bar, val in zip(bars, final_weights):
        x = val + 0.01 if val >= 0 else val - 0.05
        ax.text(x, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", fontsize=9)

    # ── 图4：IC值 ─────────────────────────────────────────────
    ax = axes[1, 1]
    ic_vals   = [ic_values.get(f, 0) for f in factors]
    ic_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ic_vals]
    ax.barh(factors, ic_vals, color=ic_colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline( 0.03, color="green", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(-0.03, color="green", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("因子 IC 值\n（虚线=±0.03有效阈值）")
    ax.set_xlabel("IC")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  图表已保存：{save_path}")


# ══════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  随机森林因子权重优化")
    print("=" * 60)

    # ── 1. 加载数据 ───────────────────────────────────────────
    panel = load_panel()

    # ── 2. 截面Rank标准化 ─────────────────────────────────────
    avail_factors = [f for f in CANDIDATE_FACTORS if f in panel.columns]
    panel = cross_section_rank(panel, avail_factors)
    rank_cols = [f"RANK_{f}" for f in avail_factors if f"RANK_{f}" in panel.columns]

    # ── 3. 清理数据 ───────────────────────────────────────────
    model_cols = rank_cols + [TARGET_COL, "trade_date"]
    df_clean = panel[model_cols].dropna()
    print(f"\n  清理后样本：{len(df_clean)} 行")

    X = df_clean[rank_cols].values
    y = df_clean[TARGET_COL].values

    # ── 4. 时序交叉验证 ───────────────────────────────────────
    print("\n  时序交叉验证（5折）...")
    splits = time_series_cv_split(df_clean["trade_date"], n_splits=5)
    cv_scores_rf  = []
    cv_scores_gbm = []

    for i, (train_dates, test_dates) in enumerate(splits):
        train_mask = df_clean["trade_date"].isin(train_dates)
        test_mask  = df_clean["trade_date"].isin(test_dates)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(X_train) < 100 or len(X_test) < 50:
            continue

        rf_fold  = train_rf(X_train, y_train)
        gbm_fold = train_gbm(X_train, y_train)

        score_rf  = r2_score(y_test, rf_fold.predict(X_test))
        score_gbm = r2_score(y_test, gbm_fold.predict(X_test))
        cv_scores_rf.append(score_rf)
        cv_scores_gbm.append(score_gbm)
        print(f"    Fold {i+1}:  RF R²={score_rf:.4f}  GBM R²={score_gbm:.4f}")

    print(f"  RF  平均R²: {np.mean(cv_scores_rf):.4f} ± {np.std(cv_scores_rf):.4f}")
    print(f"  GBM 平均R²: {np.mean(cv_scores_gbm):.4f} ± {np.std(cv_scores_gbm):.4f}")

    # ── 5. 全量训练获取特征重要性 ─────────────────────────────
    print("\n  全量训练，获取特征重要性...")
    rf_full  = train_rf(X, y)
    gbm_full = train_gbm(X, y)

    factor_names = [f.replace("RANK_", "") for f in rank_cols]
    importances_rf  = dict(zip(factor_names, rf_full.feature_importances_))
    importances_gbm = dict(zip(factor_names, gbm_full.feature_importances_))

    print("\n  RF特征重要性：")
    for f, imp in sorted(importances_rf.items(), key=lambda x: -x[1]):
        print(f"    {f:20s}  {imp:.4f}")

    # ── 6. 因子方向分析 ───────────────────────────────────────
    directions, ic_values = calc_factor_directions(
        panel, rank_cols, TARGET_COL
    )

    # ── 7. 计算最优权重 ───────────────────────────────────────
    weights = calc_optimal_weights(
        importances_rf, directions, ic_values, threshold=0.02
    )

    print("\n" + "=" * 60)
    print("  最优因子权重（直接复制到 02_factor_calc.py）")
    print("=" * 60)
    print("\n  score = (")
    for factor, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        if w == 0:
            continue
        sign = "+" if w >= 0 else ""
        rank_f = f'cs_rank_zscore("{factor}")'
        print(f"    {sign}{w:.4f} * {rank_f}")
    print("  )")

    # ── 8. 有效因子汇总 ───────────────────────────────────────
    print("\n  因子有效性汇总：")
    print(f"  {'因子':<20} {'IC':>8} {'RF重要性':>10} {'最终权重':>10} {'状态':>8}")
    print("  " + "-" * 60)
    for f in avail_factors:
        ic  = ic_values.get(f, 0)
        imp = importances_rf.get(f, 0)
        w   = weights.get(f, 0)
        status = "✅ 保留" if abs(w) > 0.05 else ("⚠ 降权" if abs(w) > 0 else "❌ 剔除")
        print(f"  {f:<20} {ic:>+8.4f} {imp:>10.4f} {w:>+10.4f}  {status}")

    # ── 9. 可视化 ─────────────────────────────────────────────
    plot_results(
        importances_rf, importances_gbm, weights, ic_values,
        cv_scores_rf,
        save_path=os.path.join(REPORT_DIR, "rf_factor_weights.png")
    )

    # ── 10. 输出可直接粘贴的代码片段 ─────────────────────────
    print("\n" + "=" * 60)
    print("  ▼ 直接粘贴到 02_factor_calc.py 第五步 ▼")
    print("=" * 60)
    lines = ["        score = ("]
    valid_weights = {f: w for f, w in weights.items() if abs(w) > 0.01}
    for i, (factor, w) in enumerate(
        sorted(valid_weights.items(), key=lambda x: -abs(x[1]))
    ):
        sign    = "+" if w >= 0 else ""
        comma   = "" if i == len(valid_weights) - 1 else ""
        ic_note = f"IC={ic_values.get(factor,0):+.3f}"
        lines.append(
            f"            {sign}{w:.4f} * cs_rank_zscore(\"{factor}\")   "
            f"# {ic_note}"
        )
    lines.append("        )")
    lines.append("        panel.loc[idx, \"MULTI_SCORE\"] = score.values")
    print("\n".join(lines))
    print("=" * 60)