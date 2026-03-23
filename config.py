# ============================================================
#  config.py — 全局配置中心
#  整个项目只需要改这一个文件里的参数
# ============================================================

import os

# ── 1. Tushare Token ─────────────────────────────────────填自己的token
TUSHARE_TOKEN = 
# ── 2. 路径配置 ───────────────────────────────────────────
BASE_DIR    = r"D:\量化项目"
DATA_DIR    = os.path.join(BASE_DIR, "data")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")

# ── 3. 研究区间 ───────────────────────────────────────────
START_DATE  = "20240101"   # 数据开始日期
END_DATE    = "20260320"   # 数据结束日期

# ── 4. 持仓周期 ───────────────────────────────────────────
HOLD_PERIOD = 20           # 短线：20个交易日
# ── 5. 股票池 ─────────────────────────────────────────────
def get_stock_pool():
    # 优先尝试从 Tushare 拉取中证500成分股
    try:
        import tushare as ts
        pro = ts.pro_api(TUSHARE_TOKEN)
        df = pro.index_weight(index_code="000905.SH", trade_date="")
        if df is not None and not df.empty:
            # 取最新日期的成分股
            latest = df[df["trade_date"] == df["trade_date"].max()]
            codes = latest["con_code"].tolist()
            print(f"  股票池（中证500动态）：{len(codes)} 只")
            return codes
    except Exception as e:
        print(f"  Tushare 拉取中证500失败（{e}），回退到本地文件")

    # 回退：从本地文件读取
    try:
        with open(r"D:\量化项目\stock_pool.txt", "r") as f:
            codes = [line.strip() for line in f if line.strip()]
        print(f"  股票池（本地文件）：{len(codes)} 只")
        return codes
    except:
        return []

STOCK_POOL = get_stock_pool()
# ── 6. 风控参数 ─────────────────────
STOP_LOSS   = 0.10         # 止损 10%
TAKE_PROFIT = 0.30         # 止盈 30%
MAX_POSITION = 0.25        # 单股最大仓位 25%