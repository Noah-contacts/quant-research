# ============================================================
#  strategy_gann.py — 江恩线 + 筹码因子 组合策略
#
#  买入逻辑：
#    A. MULTI_SCORE >= SCORE_ENTRY（100% 主导选股排序）
#    B. 江恩辅助加分：回踩 1:2 或 1:1 支撑线附近（±5%）
#       → 在支撑附近：final_score = MULTI_SCORE × 1.2
#       → 未在支撑：  final_score = MULTI_SCORE × 1.0
#    C. CTRL_ALPHA（仓位微调 ±20%，不参与候选池筛选）
#
#  卖出逻辑：
#    - MULTI_SCORE 跌破 SCORE_EXIT
#    - 江恩1:1线有效跌破 + 因子同步走弱
#    - 硬止损：跌幅超过 STOP_LOSS
#    - CTRL_ALPHA 偏离买入时最优百分位区间（软性提前离场）
#
#  交易成本：单边 0.15%（含印花税、手续费）
# ============================================================

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import pandas as pd
import numpy as np
from config import DATA_DIR, STOCK_POOL

FACTOR_DIR  = os.path.join(DATA_DIR, "factors")
TOP_N       = 25  # 每次最多持仓数量（原10→25，降低集中度）
HOLD_PERIOD = 20  # 调仓周期（交易日）

# ── 参数（可调） ───────────────────────────────────────────────
# 因子文件实际分布（mv_decap 残差后）：
#   新版分布（Z-score后）：min~-2.43  25%~-0.74  50%~-0.12  75%~+0.33  90%~+0.70  max~+2.18
#   > 0.33 ≈ 截面前25%

SCORE_ENTRY     = 0.10   # 买入：截面前25%（75th=0.331）
SCORE_EXIT      = -0.12  # 卖出：跌破中位数（50th=-0.123），迟滞带~0.45个单位
GANN_SCORE_WEAK = -0.30  # 江恩破位时走弱判定（原-0.05在新分布中太靠近均值）
GANN_TOL        = 0.05   # 江恩支撑容忍范围（±5%）
RSI_HIGH        = 80     # RSI 过热线
STOP_LOSS       = 0.10   # 硬止损
GANN_SWING      = 20     # 江恩回看窗口
GANN_BEAR       = 2.0    # 压力线斜率
GANN_FAST       = 4.0    # 压力线斜率

# ── CTRL_ALPHA 软性仓位微调参数（不参与选股/Gann/卖出）──────
# 分层回测实证：第4组(20~45th百分位) = 3.298% 最优
# 仅用于 pos_weight = 0.8 + 0.4 × gate(pct)，微调持仓规模
CTRL_OPT_LO = 0.20   # 最优区间百分位下沿
CTRL_OPT_HI = 0.45   # 最优区间百分位上沿


def _ctrl_pct(ctrl_val, ctrl_series):
    """
    计算 ctrl_val 在当日截面分布中的百分位 ∈ [0,1]
    ctrl_series: 当日所有股票的 CTRL_ALPHA Series（已 dropna）
    无数据时返回 0.30（第4组中心位置，保守中性）
    """
    if pd.isna(ctrl_val) or len(ctrl_series) < 10:
        return 0.30
    return float((ctrl_series < float(ctrl_val)).mean())


def ctrl_gate_pct(pct):
    """软性权重 ∈ [0,1]，最优区间(20~45th)→1.0，两侧线性衰减，无硬过滤"""
    if CTRL_OPT_LO <= pct <= CTRL_OPT_HI:
        return 1.0
    if pct < CTRL_OPT_LO:
        return max(0.0, pct / CTRL_OPT_LO)
    else:
        return max(0.0, 1.0 - (pct - CTRL_OPT_HI) / (1.0 - CTRL_OPT_HI))


def ctrl_position_weight(pct):
    """pos_weight ∈ [0.8, 1.2]，最优区间超配20%，偏离低配20%"""
    return 0.8 + 0.4 * ctrl_gate_pct(pct)

# 只信任 MULTI_SCORE（Rank残差，新版），排除旧版 COMBO_SCORE
TRUSTED_SCORE_COL = "MULTI_SCORE"


# ── 因子数据加载 ───────────────────────────────────────────────
def load_factor_data():
    """一次性加载所有股票因子数据，只加载含 MULTI_SCORE 的文件"""
    cache = {}
    skipped = 0
    for code in STOCK_POOL:
        path = os.path.join(FACTOR_DIR, f"{code}_factors.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            # 只信任新版 MULTI_SCORE 文件，跳过旧版 COMBO_SCORE
            if TRUSTED_SCORE_COL not in df.columns:
                skipped += 1
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).str[:10])
            df = df.set_index("trade_date").sort_index()
            if SCORE_COL is None:
                _detect_score_col(df)
            cache[code] = df
        except Exception:
            continue
    print(f"  因子缓存：{len(cache)} 只（跳过旧版COMBO_SCORE文件 {skipped} 只）")
    return cache


SCORE_COL = None  # 运行时自动检测："MULTI_SCORE" 或 "COMBO_SCORE"

def _detect_score_col(df):
    """自动检测因子列名"""
    global SCORE_COL
    if SCORE_COL is not None:
        return SCORE_COL
    for col in ["MULTI_SCORE", "COMBO_SCORE"]:
        if col in df.columns:
            SCORE_COL = col
            print(f"  [因子列] 自动检测到：{col}")
            return col
    SCORE_COL = "MULTI_SCORE"
    return SCORE_COL


def get_factor_row(cache, code, trade_date):
    """获取某只股票在某日的因子行，返回 Series 或 None"""
    if code not in cache:
        return None
    df = cache[code]
    if SCORE_COL is None:
        _detect_score_col(df)
    ts = pd.Timestamp(trade_date)
    if ts not in df.index:
        return None
    return df.loc[ts]


def get_score(row):
    """从因子行读取 MULTI_SCORE（新版Rank残差，不使用旧版COMBO_SCORE）"""
    if row is None:
        return np.nan
    v = row.get(TRUSTED_SCORE_COL, np.nan)
    return float(v) if not pd.isna(v) else np.nan


# ── 江恩线计算 ─────────────────────────────────────────────────
def calc_gann_support(df, trade_date, slope=1.0, window=GANN_SWING):
    """
    江恩上升支撑线：
    取过去 window 根 K 线的最低点为波段起点，
    支撑价 = 低点价格 + 日历天数 * 0.01 * slope
    slope=1.0 → 1:1 线，slope=0.5 → 1:2 线
    """
    ts = pd.Timestamp(trade_date)
    if ts not in df.index:
        return None
    idx = df.index.get_loc(ts)
    if idx < window:
        return None
    seg = df.iloc[idx - window: idx]
    low_price = seg["low"].min()
    low_date  = seg["low"].idxmin()
    days_since = (ts - low_date).days
    return low_price + days_since * 0.01 * slope


def calc_gann_resistance(df, trade_date, slope=GANN_BEAR, window=GANN_SWING):
    """
    江恩压力线：
    以同一波段低点为起点，slope 越大压力越强
    slope=2.0 → 2:1 线，slope=4.0 → 4:1 线
    """
    return calc_gann_support(df, trade_date, slope=slope, window=window)


def near_gann_support(df, trade_date, close):
    """
    江恩支撑确认：
    股价必须满足两个条件才算有效支撑回踩：
      1. close > support（在支撑线上方，确认上升趋势）
      2. close < support * (1 + GANN_TOL)（回踩至支撑线附近，不超过10%）
    这样只买"上升趋势中回调到支撑"的股票，排除正在跌破支撑的股票
    """
    for slope in [0.5, 1.0]:   # 先检查1:2，再检查1:1（斜率越小越保守）
        sup = calc_gann_support(df, trade_date, slope=slope)
        if sup is None or sup <= 0:
            continue
        # 在支撑上方 且 距支撑不超过 GANN_TOL
        if sup < close <= sup * (1 + GANN_TOL):
            return True, sup
    return False, None


# ── Backtrader 策略类 ──────────────────────────────────────────
class GannFactorStrategyV2(bt.Strategy):

    def __init__(self):
        self.day_count    = 0
        self.order_list   = []
        self.entry_price  = {}   # {code: 买入价格}
        self.entry_score  = {}   # {code: 买入时 MULTI_SCORE}
        self.entry_ctrl   = {}   # {code: 买入时 CTRL_ALPHA 截面百分位}
        self._ctrl_series_today = pd.Series(dtype=float)  # 每日截面CTRL_ALPHA，供百分位计算
        self.factor_cache = load_factor_data()
        print(f"  因子缓存加载完毕，共 {len(self.factor_cache)} 只股票")

    # ── A. 因子层筛选 ──────────────────────────────────────────
    def _factor_pass(self, row):
        """评分高=好股票"""
        score = get_score(row)
        if pd.isna(score) or score < SCORE_ENTRY:
            return False
        return True

    # ── B. 江恩加分（辅助，不作硬性门槛）──────────────────────
    def _gann_bonus(self, code, trade_date, close):
        if code not in self.factor_cache:
            return 1.0, None
        df = self.factor_cache[code]
        hit, sup = near_gann_support(df, trade_date, close)
        return (1.2, sup) if hit else (1.0, None)

    # ── 综合买入评分 ──────────────────────────────────────────
    def _buy_score(self, code, trade_date, close, ctrl_series):
        """
        ctrl_series: 当日截面所有股票的 CTRL_ALPHA 值（pd.Series，已 dropna）
                     用于计算百分位 gate，确保过滤阈值自适应市场状态
        """
        row = get_factor_row(self.factor_cache, code, trade_date)
        if row is None:
            return -999, None, 0.6
        if not self._factor_pass(row):
            return -999, None, 0.6

        ctrl_val = row.get("CTRL_ALPHA", np.nan)

        # ── 计算截面百分位（自适应市场状态）────────────────────
        pct = _ctrl_pct(ctrl_val, ctrl_series)

        score           = get_score(row)
        gann_bonus, sup = self._gann_bonus(code, trade_date, close)
        final_score     = score * gann_bonus   # MULTI_SCORE 主导，Gann 轻微加分

        # ── CTRL_ALPHA：仓位微调 ±20%，不影响排序得分 ──────────
        position_weight = ctrl_position_weight(pct)

        return final_score, sup, position_weight

    # ── 卖出信号判断 ──────────────────────────────────────────
    def _should_sell(self, code, trade_date, data_feed):
        row   = get_factor_row(self.factor_cache, code, trade_date)
        close = data_feed.close[0]

        # ── 硬止损：价格跌幅超过阈值 ──────────────────────────
        entry = self.entry_price.get(code, close)
        if entry > 0 and (close - entry) / entry < -STOP_LOSS:
            return True, f"硬止损：跌幅>{STOP_LOSS*100:.0f}%"

        if row is None:
            return False, ""

        score = get_score(row)

        # ── 因子失效：评分跌破 SCORE_EXIT ──────────────────────
        if not pd.isna(score) and score < SCORE_EXIT:
            return True, f"卖出：因子失效 score={score:.3f}"

        # ── 止损：江恩1:1线有效跌破 + 因子同步走弱 ───────────
        if code in self.factor_cache:
            df  = self.factor_cache[code]
            sup = calc_gann_support(df, trade_date, slope=1.0)
            if sup is not None and close < sup * 0.97:
                if not pd.isna(score) and score < GANN_SCORE_WEAK:
                    return True, f"止损：江恩破位+因子走弱 支撑={sup:.2f}"

        return False, ""

    # ── 换仓时：已持仓股也参与评分竞争 ───────────────────────
    def _should_replace(self, code, trade_date, close):
        """
        评估当前持仓股的保留得分，用于与新候选股竞争排名。
        若得分不如新候选则替换，实现"滚动优胜劣汰"。
        """
        row = get_factor_row(self.factor_cache, code, trade_date)
        if row is None:
            return -999
        score = get_score(row)
        if pd.isna(score):
            return -999
        bonus, _ = self._gann_bonus(code, trade_date, close)
        return float(score) * bonus

    # ── 每日执行 ──────────────────────────────────────────────
    def next(self):
        self.day_count += 1
        current_date = self.datas[0].datetime.date(0)

        # ── 更新当日截面 CTRL_ALPHA 分布（供百分位gate使用）──
        ctrl_vals = []
        for data in self.datas:
            row = get_factor_row(self.factor_cache, data._name, current_date)
            if row is not None:
                v = row.get("CTRL_ALPHA", np.nan)
                if not pd.isna(v):
                    ctrl_vals.append(float(v))
        self._ctrl_series_today = pd.Series(ctrl_vals, dtype=float)

        # ── 每日检查卖出/止损 ──────────────────────────────────
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size <= 0:
                continue
            sell_flag, reason = self._should_sell(data._name, current_date, data)
            if sell_flag:
                o = self.close(data)
                if o:
                    self.order_list.append(o)
                    print(f"  [{current_date}] 卖出 {data._name}  原因:{reason}")

        # ── 每 HOLD_PERIOD 天换仓 ─────────────────────────────
        if self.day_count % HOLD_PERIOD != 0:
            return

        # 取消未完成订单
        for o in self.order_list:
            self.cancel(o)
        self.order_list = []

        # 对所有股票（含已持仓）统一评分，实现"滚动优胜劣汰"
        all_scores = {}
        all_supports = {}
        all_ctrl   = {}   # {code: ctrl_multiplier}
        held_codes = set()

        for data in self.datas:
            code  = data._name
            close = data.close[0]
            if close <= 0 or np.isnan(close):
                continue

            if self.getposition(data).size > 0:
                # 持仓股：使用保留评分（含轻微粘滞，避免过度换手）
                held_codes.add(code)
                s = self._should_replace(code, current_date, close)
                # 因子跌破 SCORE_EXIT 的持仓股不参与保留竞争（由每日卖出逻辑处理）
                row = get_factor_row(self.factor_cache, code, current_date)
                cur_score = get_score(row)
                if s > -999 and (pd.isna(cur_score) or cur_score >= SCORE_EXIT):
                    all_scores[code]   = s + 0.02
                    all_supports[code] = None
            else:
                # 候选股：正常评分，传入截面分布供百分位gate计算
                score, sup, pos_weight = self._buy_score(
                    code, current_date, close, self._ctrl_series_today
                )
                if score > -999:
                    all_scores[code]   = score
                    all_supports[code] = sup
                    all_ctrl[code]     = pos_weight

        # 全量排名，选前 TOP_N
        target_set = set(sorted(all_scores, key=all_scores.get, reverse=True)[:TOP_N])
        if not target_set:
            return

        # 卖出不在 target_set 中的持仓
        for data in self.datas:
            code = data._name
            if self.getposition(data).size > 0 and code not in target_set:
                o = self.close(data)
                if o:
                    self.order_list.append(o)
                    score_now = all_scores.get(code, -999)
                    print(f"  [{current_date}] 换仓卖出 {code}  得分:{score_now:.3f}")

        # 买入 target_set 中未持仓的股票
        new_buys = target_set - held_codes
        if not new_buys:
            return

        print(f"  [{current_date}] 目标{len(target_set)}只  新买入{len(new_buys)}只  {sorted(new_buys)}")

        portfolio_value = self.broker.getvalue()
        per_stock = portfolio_value * 0.95 / TOP_N
        per_stock_max = portfolio_value * 1.5 / TOP_N  # 单股仓位上限

        for code in sorted(new_buys):
            try:
                data = self.getdatabyname(code)
            except Exception:
                continue

            price      = data.close[0]
            pos_weight = all_ctrl.get(code, 1.0)
            base       = min(per_stock * pos_weight, per_stock_max)
            size       = int(base / price / 100) * 100
            cash       = self.broker.getcash()
            if cash < price * size:
                size = int(cash * 0.95 / price / 100) * 100
            if size < 100:
                continue

            o = self.buy(data=data, size=size)
            if o:
                self.order_list.append(o)
                self.entry_price[code] = price
                self.entry_score[code] = all_scores.get(code, 0)
                # 记录买入时的截面百分位（用于离场判断，不是pos_weight）
                row = get_factor_row(self.factor_cache, code, current_date)
                entry_ctrl_val = row.get("CTRL_ALPHA", np.nan) if row is not None else np.nan
                self.entry_ctrl[code] = _ctrl_pct(entry_ctrl_val, self._ctrl_series_today)
                sup_str = f"{all_supports[code]:.2f}" if all_supports.get(code) else "N/A"
                print(f"    买入 {code}  价:{price:.2f}  量:{size}"
                      f"  得分:{all_scores.get(code,0):.3f}  支撑:{sup_str}"
                      f"  pos_weight:{pos_weight:.2f}")

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price[order.data._name] = order.executed.price
                print(f"  成交(买) {order.data._name} "
                      f"价格:{order.executed.price:.2f} 数量:{order.executed.size}")
            else:
                print(f"  成交(卖) {order.data._name} "
                      f"价格:{order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass

    def notify_trade(self, trade):
        if trade.isclosed:
            pct = (trade.pnlcomm / abs(trade.price * trade.size) * 100
                   if trade.size != 0 else 0)
            print(f"  平仓 {trade.data._name}  盈亏:{trade.pnlcomm:+.0f}  收益率:{pct:+.2f}%")