# ============================================================
#  run_backtest_gann.py — 江恩线+筹码因子策略 回测入口
#
#  运行：python backtest/run_backtest_gann.py
# ============================================================

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import DATA_DIR, STOCK_POOL, REPORT_DIR, START_DATE, END_DATE
from backtest.strategy_gann import GannFactorStrategyV2
os.makedirs(REPORT_DIR, exist_ok=True)
COMMISSION = 0.0015   # 单边0.15%（含印花税）


def load_data(code, start, end):
    path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
        df = df.set_index("trade_date").sort_index()
        start_dt = pd.to_datetime(str(start), format="%Y%m%d")
        end_dt   = pd.to_datetime(str(end),   format="%Y%m%d")
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        # 过滤起始日太晚的股票（数据行数不足 or 截断后起始日超过 START_DATE+6个月）
        start_dt_limit = start_dt + pd.DateOffset(months=6)
        if len(df) < 60 or df.index.min() > start_dt_limit:
            return None
        df = df.rename(columns={"vol": "volume"})
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return None
        return bt.feeds.PandasData(
            dataname=df,
            open="open", high="high", low="low",
            close="close", volume="volume",
            openinterest=-1,
        )
    except:
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("   江恩线 + 筹码因子 策略回测")
    print(f"   区间：{START_DATE} → {END_DATE}")
    print(f"   股票池：{len(STOCK_POOL)} 只")
    print(f"   交易成本：单边 {COMMISSION*100:.2f}%")
    print("=" * 60)

    cerebro = bt.Cerebro()

    loaded = 0
    for code in STOCK_POOL:
        data = load_data(code, START_DATE, END_DATE)
        if data is not None:
            cerebro.adddata(data, name=code)
            loaded += 1

    print(f"  成功加载 {loaded} 只股票数据")
    if loaded == 0:
        print("❌ 没有数据，退出")
        sys.exit(1)

    cerebro.addstrategy(GannFactorStrategyV2)

    cerebro.broker.setcash(1_000_000)
    cerebro.broker.setcommission(commission=COMMISSION)

    # 分析指标
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                        _name="sharpe",
                        riskfreerate=0.03,
                        annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn,  _name="annual")
    cerebro.addanalyzer(bt.analyzers.TimeReturn,    _name="timereturn")

    print("\n  开始回测...\n")
    init_cash  = cerebro.broker.getvalue()
    results    = cerebro.run()
    final_cash = cerebro.broker.getvalue()
    strat      = results[0]

    # ── 计算指标 ──────────────────────────────────────────────
    total_ret   = (final_cash - init_cash) / init_cash * 100
    sharpe_raw  = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    sharpe      = sharpe_raw if sharpe_raw is not None else float("nan")
    dd_info     = strat.analyzers.drawdown.get_analysis()
    max_dd      = dd_info.get("max", {}).get("drawdown", 0)
    trade_info  = strat.analyzers.trades.get_analysis()
    total_trades= trade_info.get("total", {}).get("closed", 0)
    won_trades  = trade_info.get("won",   {}).get("total",  0)
    win_rate    = won_trades / total_trades * 100 if total_trades > 0 else 0

    # 年化收益率（用 TimeReturn 手动算）
    time_ret    = strat.analyzers.timereturn.get_analysis()
    if time_ret:
        daily_rets  = np.array(list(time_ret.values()))
        ann_ret     = (np.prod(1 + daily_rets) ** (252 / len(daily_rets)) - 1) * 100
        volatility  = np.std(daily_rets) * np.sqrt(252) * 100
    else:
        ann_ret     = 0.0
        volatility  = 0.0

    # Calmar 比率
    calmar = ann_ret / max_dd if max_dd > 0 else float("nan")

    print("\n" + "=" * 60)
    print("  江恩线 + 筹码因子策略 — 回测结果")
    print("=" * 60)
    print(f"  初始资金：        CNY {init_cash:>12,.0f}")
    print(f"  最终资金：        CNY {final_cash:>12,.0f}")
    print(f"  总收益率：        {total_ret:>+10.2f}%")
    print(f"  年化收益率：      {ann_ret:>+10.2f}%")
    print(f"  年化波动率：      {volatility:>10.2f}%")
    print(f"  夏普比率：        {sharpe:>10.3f}")
    print(f"  最大回撤：        {max_dd:>10.2f}%")
    print(f"  Calmar比率：      {calmar:>10.3f}")
    print(f"  总交易次数：      {total_trades:>10}")
    print(f"  胜率：            {win_rate:>10.1f}%")
    print("=" * 60)

    # 年度分析
    annual_info = strat.analyzers.annual.get_analysis()
    if annual_info:
        print("\n  年度收益率：")
        for year, ret in sorted(annual_info.items()):
            print(f"    {year}：{ret*100:+.2f}%")

    # 综合评价
    print()
    if total_ret > 0 and sharpe > 1.0:
        print("  ✅ 策略表现优秀（正收益 + 夏普>1）")
    elif total_ret > 0 and sharpe > 0.5:
        print("  ⚠️  策略基本可用，建议优化因子阈值")
    elif total_ret > 0:
        print("  ⚠️  有正收益但夏普较低，风险调整后表现一般")
    else:
        print("  ❌ 策略亏损，建议调整因子阈值或买卖逻辑")

    # ── 保存净值曲线图 ────────────────────────────────────────
    try:
        time_ret = strat.analyzers.timereturn.get_analysis()
        dates = list(time_ret.keys())
        rets  = list(time_ret.values())
        # 在序列前插入 1.0 作为初始净值，确保曲线从 1.0 起步
        nav_series = pd.Series([0.0] + rets,
                               index=[pd.to_datetime(dates[0]) - pd.Timedelta(days=1)] + pd.to_datetime(dates).tolist())
        nav = nav_series.add(1).cumprod()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # 净值曲线
        axes[0].plot(nav.index, nav.values, color="steelblue", linewidth=1.5, label="策略净值")
        axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_title(f"江恩线+筹码因子策略 净值曲线\n"
                          f"总收益:{total_ret:+.2f}%  夏普:{sharpe:.3f}  最大回撤:{max_dd:.2f}%",
                          fontsize=12)
        axes[0].set_ylabel("净值")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 回撤曲线
        drawdown_curve = (nav / nav.cummax() - 1) * 100
        axes[1].fill_between(drawdown_curve.index, drawdown_curve.values, 0,
                             color="salmon", alpha=0.6, label="回撤")
        axes[1].set_title("回撤曲线")
        axes[1].set_ylabel("回撤 (%)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(REPORT_DIR, "backtest_gann.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  净值曲线已保存：{chart_path}")
    except Exception as e:
        print(f"  图表保存失败：{e}")

    print(f"  报告目录：{REPORT_DIR}")
