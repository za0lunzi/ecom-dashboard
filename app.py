import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Ecom Dashboard", layout="wide")
st.title("Ecom Dashboard 🚀")

SALES_FILE = r"data/sales.xlsx"
MAP_FILE   = r"data/mapping.xlsx"

# ===== 可视化格式偏好 =====
HOVER_FONT_SIZE = 18
AXIS_TICK_FORMAT = ",.0f"   # 千分位 + 0小数（不要k、不要小数）
PCT_FORMAT = ".1%"          # 百分比显示

COST_COLS = ["采购费", "海运费", "佣金", "配送费", "广告费", "仓储费"]


@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

def norm_platform(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def make_product_key(platform: str, sku: str) -> str:
    return f"{platform}|{sku}"

def safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def wow_delta(s: pd.Series) -> pd.Series:
    return s - s.shift(1)

def wow_pct_traditional(s: pd.Series) -> pd.Series:
    """环比% = (本周-上周)/上周（传统口径）"""
    prev = s.shift(1)
    return np.where(prev.notna() & (prev != 0), (s - prev) / prev, np.nan)

# ====== 安全格式化（不会因为str报错）======
def fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:,.0f}"

def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.1%}"

def format_series_for_table(s: pd.Series, typ: str) -> pd.Series:
    if typ == "pct":
        return s.apply(fmt_pct)
    return s.apply(fmt_money)

def _is_pct_col(col_name: str) -> bool:
    return col_name in ["利润率"]


# ========= 读取 =========
sales = load_excel(SALES_FILE)
mp = load_excel(MAP_FILE)

# ========= 清理 =========
sales = sales.loc[:, ~sales.columns.astype(str).str.startswith("Unnamed")].copy()
sales["平台"] = sales["平台"].apply(norm_platform)
sales["VENDOR_SKU"] = sales["VENDOR_SKU"].astype(str).str.strip()

# 日期
sales["ORDER_PLACED_DT"] = pd.to_datetime(sales["ORDER_PLACED_DT"], errors="coerce")

# year/week -> year_week（避免跨年混淆）
sales["year"] = pd.to_numeric(sales["year"], errors="coerce")
sales["week"] = pd.to_numeric(sales["week"], errors="coerce")
sales["year_week"] = sales.apply(
    lambda r: f"{int(r['year'])}-W{int(r['week']):02d}" if pd.notna(r["year"]) and pd.notna(r["week"]) else None,
    axis=1
)

# product_key
sales["product_key"] = sales.apply(lambda r: make_product_key(r["平台"], r["VENDOR_SKU"]), axis=1)

# 数值列
num_cols = ["gmv（￥）", "QUANTITY", "利润", "利润率"] + COST_COLS
sales = safe_to_numeric(sales, num_cols)

# ========= 映射表拆平台（统一字段） =========
mp = mp.copy()
mp["沃尔玛SKU"] = mp["沃尔玛SKU"].astype(str).str.strip()
mp["亚马逊sku"] = mp["亚马逊sku"].astype(str).str.strip()

wm = mp[["沃尔玛SKU", "月度目标-wfs", "目前库存-wfs", "在途库存-wfs"]].copy()
wm = wm.rename(columns={
    "沃尔玛SKU": "VENDOR_SKU",
    "月度目标-wfs": "月度目标_QUANTITY",
    "目前库存-wfs": "当前库存",
    "在途库存-wfs": "在途库存",
})
wm["平台"] = "沃尔玛"
wm["VENDOR_SKU"] = wm["VENDOR_SKU"].astype(str).str.strip()
wm["product_key"] = wm.apply(lambda r: make_product_key(r["平台"], r["VENDOR_SKU"]), axis=1)

amz = mp[["亚马逊sku", "月度目标-fba", "目前库存-fba", "在途库存-fba"]].copy()
amz = amz.rename(columns={
    "亚马逊sku": "VENDOR_SKU",
    "月度目标-fba": "月度目标_QUANTITY",
    "目前库存-fba": "当前库存",
    "在途库存-fba": "在途库存",
})
amz["平台"] = "亚马逊"
amz["VENDOR_SKU"] = amz["VENDOR_SKU"].astype(str).str.strip()
amz["product_key"] = amz.apply(lambda r: make_product_key(r["平台"], r["VENDOR_SKU"]), axis=1)

mp_long = pd.concat([wm, amz], ignore_index=True)
mp_long = safe_to_numeric(mp_long, ["月度目标_QUANTITY", "当前库存", "在途库存"])

# ========= 合并（left join，尾部SKU不丢） =========
df = sales.merge(
    mp_long[["product_key", "月度目标_QUANTITY", "当前库存", "在途库存"]],
    on="product_key",
    how="left"
)

# ========= Sidebar 筛选 =========
st.sidebar.header("筛选")
hide_no_target = st.sidebar.checkbox("隐藏无目标SKU（推荐）", value=True)

platform_options = ["全部"] + sorted([x for x in df["平台"].dropna().unique().tolist() if x != ""])
platform_sel = st.sidebar.selectbox("平台", platform_options, index=0)

cat_options = ["全部"] + sorted(df["产品类别"].dropna().unique().tolist())
cat_sel = st.sidebar.selectbox("产品类别", cat_options, index=0)

sku_options = ["全部"] + sorted(df["VENDOR_SKU"].dropna().unique().tolist())
sku_sel = st.sidebar.selectbox("VENDOR_SKU", sku_options, index=0)

df_f = df.copy()
if hide_no_target:
    df_f = df_f[df_f["月度目标_QUANTITY"].notna()].copy()
if platform_sel != "全部":
    df_f = df_f[df_f["平台"] == platform_sel].copy()
if cat_sel != "全部":
    df_f = df_f[df_f["产品类别"] == cat_sel].copy()
if sku_sel != "全部":
    df_f = df_f[df_f["VENDOR_SKU"] == sku_sel].copy()

st.success("✅ 数据准备完成")
c1, c2, c3, c4 = st.columns(4)
c1.metric("销售明细行数", f"{len(df):,}")
c2.metric("筛选后明细行数", f"{len(df_f):,}")
c3.metric("有目标明细行数", f"{df['月度目标_QUANTITY'].notna().sum():,}")
c4.metric("无目标明细行数", f"{df['月度目标_QUANTITY'].isna().sum():,}")

# ========= 周汇总（一次聚合出所有模块要用的列） =========
agg_dict = {
    "gmv（￥）": "sum",
    "QUANTITY": "sum",
    "利润": "sum",
    "广告费": "sum",
    "采购费": "sum",
    "海运费": "sum",
    "佣金": "sum",
    "配送费": "sum",
    "仓储费": "sum",
}
weekly = df_f.groupby("year_week", as_index=False).agg(agg_dict)
weekly["利润率"] = np.where(weekly["gmv（￥）"] != 0, weekly["利润"] / weekly["gmv（￥）"], np.nan)

# 排序
weekly["year_num"] = weekly["year_week"].str.extract(r"(\d{4})").astype(float)
weekly["week_num"] = weekly["year_week"].str.extract(r"W(\d{2})").astype(float)
weekly = weekly.sort_values(["year_num", "week_num"]).drop(columns=["year_num", "week_num"])

# ========= 1) 周趋势模块 =========
st.subheader("周趋势")

metric_map = {
    "GMV（￥）": "gmv（￥）",
    "销量（QUANTITY）": "QUANTITY",
    "利润": "利润",
    "广告费": "广告费",
    "利润率": "利润率",
}

st.sidebar.subheader("趋势图")
chart_mode = st.sidebar.selectbox("趋势模式", ["单指标", "双指标对比（双轴）"], index=0)

if chart_mode == "单指标":
    metric_label = st.sidebar.selectbox("趋势指标", list(metric_map.keys()), index=0)
    metric_col = metric_map[metric_label]

    fig = px.line(weekly, x="year_week", y=metric_col, markers=True, title=f"{metric_label}（按周）")
    if metric_col == "利润率":
        fig.update_yaxes(tickformat=PCT_FORMAT)
        fig.update_traces(hovertemplate="%{x}<br>" + f"{metric_label}: %{{y:{PCT_FORMAT}}}<extra></extra>")
    else:
        fig.update_yaxes(tickformat=AXIS_TICK_FORMAT)
        fig.update_traces(hovertemplate="%{x}<br>" + f"{metric_label}: %{{y:{AXIS_TICK_FORMAT}}}<extra></extra>")

    fig.update_layout(hoverlabel=dict(font_size=HOVER_FONT_SIZE), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ===== 趋势图下方：数据表 + 环比（只跟随当前指标）=====
    with st.expander("查看趋势数据表（含环比）", expanded=True):
        typ = "pct" if _is_pct_col(metric_col) else "money"
        t = weekly[["year_week", metric_col]].copy()
        t[f"{metric_label}_WoWΔ"] = wow_delta(t[metric_col])
        t[f"{metric_label}_WoW%"] = wow_pct_traditional(t[metric_col])

        show = pd.DataFrame()
        show["year_week"] = t["year_week"]
        show[metric_label] = format_series_for_table(t[metric_col], typ)
        show[f"{metric_label} 环比Δ"] = format_series_for_table(t[f"{metric_label}_WoWΔ"], typ)
        show[f"{metric_label} 环比%"] = format_series_for_table(t[f"{metric_label}_WoW%"], "pct")
        st.dataframe(show, use_container_width=True)

else:
    left_label = st.sidebar.selectbox("左轴指标", list(metric_map.keys()), index=0)
    right_default_idx = list(metric_map.keys()).index("利润") if "利润" in metric_map else 0
    right_label = st.sidebar.selectbox("右轴指标", list(metric_map.keys()), index=right_default_idx)

    left_col = metric_map[left_label]
    right_col = metric_map[right_label]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=weekly["year_week"], y=weekly[left_col],
            name=left_label, mode="lines+markers",
            hovertemplate="%{x}<br>" + (
                f"{left_label}: %{{y:{PCT_FORMAT}}}<extra></extra>" if _is_pct_col(left_col)
                else f"{left_label}: %{{y:{AXIS_TICK_FORMAT}}}<extra></extra>"
            )
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=weekly["year_week"], y=weekly[right_col],
            name=right_label, mode="lines+markers",
            hovertemplate="%{x}<br>" + (
                f"{right_label}: %{{y:{PCT_FORMAT}}}<extra></extra>" if _is_pct_col(right_col)
                else f"{right_label}: %{{y:{AXIS_TICK_FORMAT}}}<extra></extra>"
            )
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=f"{left_label} vs {right_label}（按周）",
        hoverlabel=dict(font_size=HOVER_FONT_SIZE),
        hovermode="x unified",
        legend_title_text=""
    )

    if _is_pct_col(left_col):
        fig.update_yaxes(tickformat=PCT_FORMAT, secondary_y=False, title_text=left_label)
    else:
        fig.update_yaxes(tickformat=AXIS_TICK_FORMAT, secondary_y=False, title_text=left_label)

    if _is_pct_col(right_col):
        fig.update_yaxes(tickformat=PCT_FORMAT, secondary_y=True, title_text=right_label)
    else:
        fig.update_yaxes(tickformat=AXIS_TICK_FORMAT, secondary_y=True, title_text=right_label)

    st.plotly_chart(fig, use_container_width=True)

    # ===== 趋势图下方：双指标数据表 + 环比（只跟随当前双指标）=====
    with st.expander("查看趋势数据表（含环比）", expanded=True):
        def _typ(col):
            return "pct" if _is_pct_col(col) else "money"

        t = weekly[["year_week", left_col, right_col]].copy()

        t[f"{left_label}_WoWΔ"] = wow_delta(t[left_col])
        t[f"{left_label}_WoW%"] = wow_pct_traditional(t[left_col])

        t[f"{right_label}_WoWΔ"] = wow_delta(t[right_col])
        t[f"{right_label}_WoW%"] = wow_pct_traditional(t[right_col])

        show = pd.DataFrame()
        show["year_week"] = t["year_week"]

        show[left_label] = format_series_for_table(t[left_col], _typ(left_col))
        show[f"{left_label} 环比Δ"] = format_series_for_table(t[f"{left_label}_WoWΔ"], _typ(left_col))
        show[f"{left_label} 环比%"] = format_series_for_table(t[f"{left_label}_WoW%"], "pct")

        show[right_label] = format_series_for_table(t[right_col], _typ(right_col))
        show[f"{right_label} 环比Δ"] = format_series_for_table(t[f"{right_label}_WoWΔ"], _typ(right_col))
        show[f"{right_label} 环比%"] = format_series_for_table(t[f"{right_label}_WoW%"], "pct")

        st.dataframe(show, use_container_width=True)

# ========= 2) 目标模块（本月目标 vs 实际） =========
st.subheader("目标与达成（按月）")

if df_f["ORDER_PLACED_DT"].notna().any():
    current_month = df_f["ORDER_PLACED_DT"].max().to_period("M")
    df_f = df_f.copy()
    df_f["year_month"] = df_f["ORDER_PLACED_DT"].dt.to_period("M")
    cur = df_f[df_f["year_month"] == current_month].copy()

    sku_month = cur.groupby(["平台", "VENDOR_SKU"], as_index=False).agg(
        本月销量=("QUANTITY", "sum"),
        月度目标=("月度目标_QUANTITY", "first"),
    )
    sku_month["完成率"] = np.where(sku_month["月度目标"] > 0, sku_month["本月销量"] / sku_month["月度目标"], np.nan)

    total_actual = sku_month["本月销量"].sum()
    total_target = sku_month["月度目标"].sum()
    total_rate = (total_actual / total_target) if total_target else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("当前月", str(current_month))
    k2.metric("本月累计销量", f"{total_actual:,.0f}")
    k3.metric("本月目标完成率", f"{total_rate:.1%}" if pd.notna(total_rate) else "—")

    st.write("SKU 目标完成率（Top 20）")
    show_top = sku_month.sort_values("完成率", ascending=False).head(20)
    st.dataframe(show_top, use_container_width=True)
else:
    st.info("当前筛选数据没有可用日期，无法计算本月目标达成。")

# ========= 3) 成本结构模块 =========
st.subheader("成本结构（按周）")

tab1, tab2 = st.tabs(["成本金额堆叠", "成本占比（100%堆叠）"])

cost_long = weekly.melt(
    id_vars=["year_week"],
    value_vars=COST_COLS,
    var_name="成本项",
    value_name="金额"
)

with tab1:
    fig_cost_amt = px.bar(
        cost_long, x="year_week", y="金额", color="成本项",
        title="每周成本结构（金额堆叠）"
    )
    fig_cost_amt.update_layout(hoverlabel=dict(font_size=HOVER_FONT_SIZE), hovermode="x unified")
    fig_cost_amt.update_yaxes(tickformat=AXIS_TICK_FORMAT)
    fig_cost_amt.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:,.0f}<extra></extra>")
    st.plotly_chart(fig_cost_amt, use_container_width=True)

with tab2:
    cost_share = cost_long.copy()
    total_by_week = cost_share.groupby("year_week")["金额"].transform("sum")
    cost_share["占比"] = np.where(total_by_week != 0, cost_share["金额"] / total_by_week, np.nan)

    fig_cost_pct = px.bar(
        cost_share, x="year_week", y="占比", color="成本项",
        title="每周成本结构（占比 100%堆叠）"
    )
    fig_cost_pct.update_layout(hoverlabel=dict(font_size=HOVER_FONT_SIZE), hovermode="x unified")
    fig_cost_pct.update_yaxes(tickformat=PCT_FORMAT)
    fig_cost_pct.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:.1%}<extra></extra>")
    st.plotly_chart(fig_cost_pct, use_container_width=True)

# ========= 4) 单周快照（金额+占比+瀑布） =========
st.subheader("单周快照（金额 + 占比 + 利润）")

week_options = weekly["year_week"].dropna().unique().tolist()
if len(week_options) == 0:
    st.info("当前筛选没有周数据。")
else:
    selected_week = st.selectbox("选择周（year_week）", week_options, index=len(week_options) - 1)
    row = weekly[weekly["year_week"] == selected_week].iloc[0]

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("GMV（￥）", f"{row['gmv（￥）']:,.0f}")
    a2.metric("利润", f"{row['利润']:,.0f}")
    a3.metric("利润率", f"{row['利润率']:.1%}" if pd.notna(row["利润率"]) else "—")
    a4.metric("广告费", f"{row['广告费']:,.0f}")

    cost_table = pd.DataFrame({
        "成本项": COST_COLS,
        "金额": [row[c] for c in COST_COLS],
    })
    cost_table["占比"] = np.where(cost_table["金额"].sum() != 0, cost_table["金额"] / cost_table["金额"].sum(), np.nan)

    left, right = st.columns([1, 1])

    with left:
        st.write("成本明细（金额 + 占比）")
        ct = cost_table.copy()
        ct["金额"] = ct["金额"].map(lambda x: f"{x:,.0f}")
        ct["占比"] = ct["占比"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        st.dataframe(ct, use_container_width=True)

    with right:
        fig_pie = px.pie(cost_table, names="成本项", values="金额", title="成本占比（环图）", hole=0.45)
        fig_pie.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,.0f}（%{percent}）<extra></extra>")
        fig_pie.update_layout(hoverlabel=dict(font_size=HOVER_FONT_SIZE))
        st.plotly_chart(fig_pie, use_container_width=True)

    # 瀑布图：GMV -> 各成本 -> 利润（成本用负数）
    measures = ["absolute"] + ["relative"] * len(COST_COLS) + ["total"]
    x = ["GMV"] + COST_COLS + ["利润"]
    y = [row["gmv（￥）"]] + [-row[c] for c in COST_COLS] + [row["利润"]]

    fig_wf = go.Figure(go.Waterfall(
        measure=measures,
        x=x,
        y=y,
        text=[f"{v:,.0f}" for v in y],
        textposition="outside",
        connector={"line": {"width": 1}},
        hovertemplate="%{x}: %{y:,.0f}<extra></extra>"
    ))
    fig_wf.update_layout(title="GMV → 成本 → 利润（瀑布图）", hoverlabel=dict(font_size=HOVER_FONT_SIZE))
    st.plotly_chart(fig_wf, use_container_width=True)

# ========= 5) 页面最下面：周报表（固定） =========
st.subheader("周报表（值 + 环比）")

# 你截图里那种：行=指标，列=各周+总计
weekly_metrics = [
    ("周GMV（￥）", "gmv（￥）", "money"),
    ("周销量", "QUANTITY", "money"),
    ("周利润", "利润", "money"),
    ("周广告费", "广告费", "money"),
    ("周利润率", "利润率", "pct"),
]

# 如果你也想把成本项放进周报表（像广告表一样）
include_costs = st.checkbox("周报表包含各成本项（采购/海运/佣金/配送/仓储）", value=True)
if include_costs:
    for c in ["采购费", "海运费", "佣金", "配送费", "仓储费"]:
        weekly_metrics.append((f"周{c}", c, "money"))

week_list = weekly["year_week"].tolist()

rows = []

# 值行
for label, col, typ in weekly_metrics:
    row = {"指标": label}
    for _, r in weekly.iterrows():
        row[r["year_week"]] = r[col]
    # 总计
    if typ == "pct" and col == "利润率":
        gsum = weekly["gmv（￥）"].sum()
        psum = weekly["利润"].sum()
        row["总计"] = (psum / gsum) if gsum else np.nan
    else:
        row["总计"] = weekly[col].sum()
    row["_type"] = typ
    rows.append(row)

# 环比行（只做 money 和 pct 都可以）
for label, col, typ in weekly_metrics:
    wow = wow_pct_traditional(weekly[col])
    row = {"指标": f"{label}环比"}
    for i, yw in enumerate(week_list):
        row[yw] = wow[i]
    row["总计"] = np.nan
    row["_type"] = "pct"
    rows.append(row)

report = pd.DataFrame(rows)

# 格式化输出
fmt = report.copy()
week_cols = [c for c in fmt.columns if c not in ["指标", "_type"]]

for idx in fmt.index:
    typ = fmt.loc[idx, "_type"]
    for c in week_cols:
        v = fmt.loc[idx, c]
        if typ == "money":
            fmt.loc[idx, c] = fmt_money(v)
        elif typ == "pct":
            fmt.loc[idx, c] = fmt_pct(v)
        else:
            fmt.loc[idx, c] = v

fmt = fmt.drop(columns=["_type"])
st.dataframe(fmt, use_container_width=True)
