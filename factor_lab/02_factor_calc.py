import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import DATA_DIR, STOCK_POOL , HOLD_PERIOD

FACTOR_DIR = os.path.join(DATA_DIR, "factors")
os.makedirs(FACTOR_DIR, exist_ok=True)


def calc_factors(df):
    """
    输入：单只股票的干净DataFrame（来自01_data_clean.py的输出）
    输出：同一个DataFrame，但多了10个因子列 + 1个目标变量列
    """

    d=df.copy()
    # ── 在 calc_factors 定义之前，顶格写 ──────────────────────────

def calc_UTR_ST(d, window=20):
    """
    UTR2.0_st：换手率结构因子
    核心逻辑：STR（换手率稳定性）与 Turn20（换手率均值）的非线性耦合
    用 softsign 映射，捕捉筹码交换中的异常溢价：
      - 高稳定 + 缩量（STR高 + Turn20低）→ 强势蓄力
      - 低稳定 + 放量（STR低 + Turn20高）→ 异常换手警示
    获利信号 = softsign(STR - Turn20_normalized) 的非线性耦合
    """
    turn = d["TURNOVER"]

    # ── Turn20：换手率20日均值（水平） ──────────────────────
    Turn20 = turn.rolling(window).mean()

    # ── STR：换手率稳定性 = 1 - CV（变异系数越小越稳定） ────
    turn_mean = turn.rolling(window).mean()
    turn_std  = turn.rolling(window).std()
    CV        = turn_std / (turn_mean + 1e-9)
    STR       = 1.0 / (1.0 + CV)          # CV→0时STR→1（极稳定），CV大时STR→0

    # ── Turn20 归一化到与STR同量纲（softsign压缩） ──────────
    Turn20_norm = Turn20 / (1.0 + Turn20.abs())   # softsign，映射到(-1,1)

    # ── 非线性耦合：STR与Turn20的差值代表"稳定-放量"背离 ────
    # 差值为正（高稳定+缩量）→ 强势蓄力 → 正信号
    # 差值为负（低稳定+放量）→ 异常换手 → 负信号
    raw_signal = STR - Turn20_norm

    # ── softsign 最终映射，抑制极值干扰 ────────────────────
    utr = raw_signal / (1.0 + raw_signal.abs())

    # ── 5日平滑降噪 ─────────────────────────────────────────
    return utr.rolling(5, min_periods=3).mean()


def calc_CTRL_ALPHA(d, window=20, year_window=252):
    """
    CTRL_ALPHA — 主升浪捕捉型时机因子

    结构：CTRL_ALPHA = Core × Gate × (0.5 + Amplifier)

    Gate      : sigmoid(UTR) ∈ (0,1)        筹码稳定性软门控
    Core      : softsign(trend_q) ∈ (-1,1)  主升浪质量主信号
    Amplifier : sigmoid(w1·A1+w2·A2)-0.5    情绪节奏+市场活跃度
                ∈ (-0.5, 0.5)

    (0.5 + Amplifier) ∈ (0, 1)              放大器不改变Core方向

    因子方向：正向（值越高=主升浪启动强度越强）
    输出范围：(-1, 1)
    """
    turn  = d["TURNOVER"]
    close = d["close"]
    pct   = d["pct_chg"]

    # ── GATE：筹码稳定性软门控 ∈ (0,1) ─────────────────────
    turn_mean   = turn.rolling(window).mean()
    turn_std    = turn.rolling(window).std()
    CV          = turn_std / (turn_mean + 1e-9)
    STR         = 1.0 / (1.0 + CV)
    Turn20_norm = turn_mean / (1.0 + turn_mean.abs())
    UTR_raw     = STR - Turn20_norm
    UTR_smooth  = UTR_raw.rolling(5, min_periods=3).mean()

    k_gate = 8.0
    gate   = 1.0 / (1.0 + np.exp(-k_gate * UTR_smooth))

    # ── CORE：主升浪质量主信号 ∈ (-1,1) ────────────────────
    ret_20  = close.pct_change(window)
    turn_20 = turn.rolling(window).mean()
    tq_raw  = ret_20 / (turn_20 + 1e-9)

    tq_mu   = tq_raw.rolling(window * 3, min_periods=window).mean()
    tq_sig  = tq_raw.rolling(window * 3, min_periods=window).std()
    tq_clip = tq_raw.clip(lower=tq_mu - 3 * tq_sig, upper=tq_mu + 3 * tq_sig)
    core    = tq_clip / (1.0 + tq_clip.abs())

    # ── AMPLIFIER：情绪节奏 + 市场活跃度 ∈ (-0.5,0.5) ──────
    big_move = (pct > 7).astype(float)
    EMD      = big_move.rolling(year_window, min_periods=60).sum()
    A1       = np.exp(-0.5 * ((EMD - 3) / 2.0) ** 2)

    turn_short = turn.rolling(5, min_periods=3).mean()
    turn_long  = turn.rolling(60, min_periods=20).mean()
    vol_acc    = turn_short / (turn_long + 1e-9) - 1.0
    A2         = vol_acc / (1.0 + vol_acc.abs())

    w1, w2       = 0.6, 0.4
    weighted_sum = w1 * A1 + w2 * A2
    amp_tanh     = np.tanh(weighted_sum)
    amp_sigmoid  = 1.0 / (1.0 + np.exp(-3.0 * amp_tanh))
    amplifier    = amp_sigmoid - 0.5

    # ── 最终合成 ─────────────────────────────────────────────
    ctrl_alpha = core * gate * (0.5 + amplifier)
    return ctrl_alpha


def calc_factors(df):
    """..."""
    d = df.copy()
    if "turnover_rate" in d.columns:
        d["TURNOVER"] = d["turnover_rate"]
    else:
        d["TURNOVER"] = d["vol"] / (d["vol"].rolling(20).mean() + 1e-9)
    
    # UTR_ST 直接调用独立函数
    d["UTR_ST"] = calc_UTR_ST(d, window=20)

    # CTRL_ALPHA：主升浪时机因子（timing，不参与选股，仅用于仓位控制）
    d["CTRL_ALPHA"] = calc_CTRL_ALPHA(d)
    
# ────────────────────────────────────────────────────
    # 因子：LWS（下影线均值）
    # 频繁长下影线 = 多次试探支撑位失败 = 弱势信号
    # 负向因子：值越大越差
    # ────────────────────────────────────────────────────
    body = abs(d["close"] - d["open"]) + 1e-9

    lower_shadow = (d[["open","close"]].min(axis=1) - d["low"])

    shadow_ratio = lower_shadow / body

    d["LWS"] = shadow_ratio.rolling(20).mean()

    # ────────────────────────────────────────────────────
    # 因子：UBL（上影线标准差）
    # 上影线波动大 = 上方抛压不稳定 = 风险信号
    # 负向因子：值越大越差
    # ────────────────────────────────────────────────────
    upper_shadow = (d["high"] - d[["open","close"]].max(axis=1))

    body = abs(d["close"] - d["open"]) + 1e-9
    ratio = upper_shadow / body

    sell_pressure = ratio > 1.5

    d["UBL"] = sell_pressure.rolling(20).mean()

   # ────────────────────────────────────────────────────
    # 因子：Neutral_MF（资金流方向）
    # 换手率 × 价格方向 = 资金净流入代理指标
    # 正值 = 资金净流入，负值 = 资金净流出
    # 市值中性化在主程序mv_decap统一处理
    # ────────────────────────────────────────────────────
    price_dir = np.sign(d["pct_chg"])
    d["Neutral_MF"] = d["TURNOVER"] * price_dir

    # ────────────────────────────────────────────────────
    # 因子：PVI_Refined（散户放量跟风）
    # 放量上涨 = 散户追涨过热 = 负向信号
    # 值越高 = 散户情绪越亢奋 = 越危险
    # ────────────────────────────────────────────────────
    vol_ratio = d["vol"] / (d["vol"].rolling(20).mean() + 1e-9)

    ret = d["pct_chg"]/100

    d["PVI_Refined"] = (vol_ratio * ret).rolling(10).sum()

    # ────────────────────────────────────────────────────
    # 因子：APBR（人气背离）
    # 换手率上升但价格涨幅下降 = 买盘在减弱
    # 值越高 = 背离越严重 = 风险越大
    # ────────────────────────────────────────────────────
    turn_change  = d["TURNOVER"].pct_change(5)
    price_change = d["pct_chg"].rolling(5).mean()
    # 换手率涨但价格涨幅缩 = 背离
    d["APBR"] = turn_change - price_change
    # 目标变量：未来5天收益率
    d["FUTURE_RET"] = d["close"].pct_change(HOLD_PERIOD).shift(-HOLD_PERIOD)
    return d

if __name__ == "__main__":
    from scipy import stats

    print("=" * 55)
    print("  因子计算 + 流水线处理")
    print("=" * 55)

    FACTORS_ALL = [
        "UTR_ST", "LWS", "UBL",
        "Neutral_MF", "PVI_Refined", "APBR",
        "CTRL_ALPHA",   # timing因子：进入Winsorize，不参与mv_decap和MULTI_SCORE
    ]

    # ── 第一步：逐只股票计算原始因子 ──────────────────────
    print("\n  第一步：计算原始因子...")
    all_frames = []
    for code in STOCK_POOL:
        csv_path = os.path.join(DATA_DIR, f"{code}.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"  {code} ...", end=" ")
        df = pd.read_csv(csv_path)
        df = calc_factors(df)
        df["ts_code"] = code
        all_frames.append(df)
        print("✅")

    if not all_frames:
        print("❌ 没有数据")
        exit()

    panel = pd.concat(all_frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(
        panel["trade_date"].astype(str).str[:10]
    )
    print(f"  合并完成：{len(panel)} 行")

# 第二步：计算截面因子 JDQS（滚动版）
    print("\n  第二步：计算截面因子 JDQS...")

    # 先按个股计算每日是否跑赢大盘
    market_avg = panel.groupby("trade_date")["pct_chg"].transform("mean")
    panel["_beat_mkt"] = (panel["pct_chg"] > market_avg).astype(float)

    # 按个股滚动20日，计算跑赢大盘的天数占比
    panel = panel.sort_values(["ts_code", "trade_date"])
    panel["JDQS"] = (
        panel.groupby("ts_code")["_beat_mkt"]
        .transform(lambda x: x.rolling(20, min_periods=10).mean())
    )
    panel.drop(columns=["_beat_mkt"], inplace=True)

    FACTORS_ALL.append("JDQS")
    print("  JDQS ✅")

   # ── 第三步：截面 Winsorize only ───────────────────────
    print("\n  第三步：截面 Winsorize 去极值...")

    factors_in_panel = [f for f in FACTORS_ALL if f in panel.columns]
    for factor in factors_in_panel:
        processed = panel[factor].copy()
        for date, group in panel.groupby("trade_date"):
            idx  = group.index
            vals = group[factor].dropna()
            if len(vals) < 10:
                continue
            # Winsorize ±3σ
            mean, std = vals.mean(), vals.std()
            vals = vals.clip(lower=mean-3*std, upper=mean+3*std)
            processed.loc[idx] = np.nan
            processed.loc[vals.index] = vals.values
        panel[factor] = processed
        print(f"    {factor} ✅")

    # ── 第四步：mv_decap 截面市值中性化 ───────────────────
    print("\n  第四步：mv_decap 市值中性化（Winsorize后做）...")

    def mv_decap(panel_df, factor):
        result = panel_df[factor].copy()
        for date, group in panel_df.groupby("trade_date"):
            data = group[[factor, "total_mv"]].dropna()
            data = data[data["total_mv"] > 0]
            if len(data) < 10:
                continue
            log_mv = np.log(data["total_mv"])
            slope, intercept, _, _, _ = stats.linregress(log_mv, data[factor])
            residual = data[factor] - (slope * log_mv + intercept)
            result.loc[residual.index] = residual.values
        return result

    # CTRL_ALPHA 是 timing 因子，结构已内含筹码稳定性，无需市值中性化
    DECAP_FACTORS = [f for f in factors_in_panel if f != "CTRL_ALPHA"]
    for factor in DECAP_FACTORS:
        if "total_mv" in panel.columns:
            print(f"    中性化：{factor} ...", end=" ")
            panel[factor] = mv_decap(panel, factor)
            print("✅")
    # 加在第五步之前，5行代码搞定
    last_date = panel["trade_date"].max()
    sample = panel[panel["trade_date"] == last_date]
    corr = sample[["UTR_ST","LWS","APBR","Neutral_MF","CTRL_ALPHA","JDQS","PVI_Refined"]].corr()
    print("\n  因子相关系数矩阵：")
    print(corr.round(3).to_string())
    print("\n  |相关系数| > 0.6 说明存在共线性问题")

   # ── 第五步：合成MULTI_SCORE（Rank→Z-score→加权）────────
    print("\n  第五步：合成 MULTI_SCORE...")
    # CTRL_ALPHA 是 timing 因子，仅用于仓位控制，不参与选股评分

    panel["MULTI_SCORE"] = np.nan
    for date, group in panel.groupby("trade_date"):
        idx = group.index

        def cs_rank_zscore(col):
            if col not in group.columns:
                return pd.Series(0.0, index=idx)
            r = group[col].rank(pct=True)
            z = (r - r.mean()) / (r.std() + 1e-9)
            return z

        # 实测方向（全量数据，高值-低值未来收益差）：
        # UTR_ST(+新版高值=稳定缩量=好), LWS(+), UBL(+), PVI(-), APBR(+), Neutral_MF(-), JDQS(-)
        score = (
            -0.50 * cs_rank_zscore("UTR_ST")        +   # 正向：低UTR=稳定缩量=好（新版UTR2.0方向已反转）
            -0.10 * cs_rank_zscore("LWS")           +   # 负向：高下影=支撑强=好
            -0.10 * cs_rank_zscore("UBL")           +   # 负向：高上影=好
            -0.10 * cs_rank_zscore("APBR")          -   # 正向：高背离=好
            -0.10 * cs_rank_zscore("PVI_Refined")   -   # 负向：高跟风=坏
            +0.20 * cs_rank_zscore("Neutral_MF")    -   # 负向：高资金流=坏
            +0.30 * cs_rank_zscore("JDQS")              # 负向：高截面涨幅=坏
        )
        panel.loc[idx, "MULTI_SCORE"] = score.values

    print("  MULTI_SCORE ✅")

    # ── 第六步：拆回每只股票保存 ───────────────────────────
    print("\n  第六步：保存因子文件...")
    success = 0
    for code in STOCK_POOL:
        stock_data = panel[panel["ts_code"] == code].copy()
        if stock_data.empty:
            continue
        save_path = os.path.join(FACTOR_DIR, f"{code}_factors.csv")
        stock_data.to_csv(save_path, index=False, encoding="utf-8-sig")
        success += 1

    print("=" * 55)
    print(f"  完成！{success} 只股票已保存")
    print(f"  保存至：{FACTOR_DIR}")
    print("=" * 55)