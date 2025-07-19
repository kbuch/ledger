# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import groupby
import os
import io
import datetime
from dateutil.relativedelta import relativedelta

def load_trade_steward(raw):
    """
    Cleanly extract only FinalTradeClosedDate and TotalNetProfitLoss
    from a messy TS export (with unquoted commas in free-text fields).
    Returns a DataFrame with exactly two columns:
      - 'Date Closed' (datetime64[ns], parsed M/D/YYYY)
      - 'P/L'         (float)
    """
    # Read all lines
    if hasattr(raw, "getvalue"):
        text = raw.getvalue().decode("utf-8", "ignore").splitlines()
    else:
        with open(raw, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().splitlines()

    header   = text[0].split(",")
    date_idx = header.index("FinalTradeClosedDate")
    pl_idx   = header.index("TotalNetProfitLoss")
    hl       = len(header)

    # how many splits to do from the right so date lands in parts[-splits]
    splits    = hl - date_idx
    pl_offset = hl - pl_idx  # so pl is in parts[-pl_offset]

    rows = []
    for line in text[1:]:
        # strip trailing commas so we don't get an empty final field
        line = line.rstrip("\n").rstrip(",")
        parts = line.rsplit(",", splits)
        # must have produced at least splits+1 parts
        if len(parts) < splits + 1:
            continue
        date_str = parts[-splits]
        pl_str   = parts[-pl_offset]
        rows.append({"Date Closed": date_str, "P/L": pl_str})

    clean = pd.DataFrame(rows)
    # parse the date & P/L
    clean["Date Closed"] = pd.to_datetime(
        clean["Date Closed"],
        format="%m/%d/%Y",
        errors="coerce"
    )
    clean["P/L"] = (
        clean["P/L"]
        .astype(str)
        .str.replace(",", "")   # drop any thousands‐sep
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    return clean.dropna(subset=["Date Closed"]).reset_index(drop=True)

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 Dashboard")

# ──────────────────────────────────────────────────────────────────────
# 📁 Data Manager: shared upload + save/load/delete
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

if "trade_bytes" not in st.session_state:
    st.session_state.trade_bytes = None
    st.session_state.trade_name  = None

st.sidebar.header("📁 Data Manager")
file_uploader = st.sidebar.file_uploader(
    "Upload trade-log CSV", type="csv", key="trade_uploader"
)
if file_uploader is not None:
    st.session_state.trade_bytes = file_uploader.getvalue()
    st.session_state.trade_name  = file_uploader.name

if st.session_state.trade_name:
    st.sidebar.markdown(f"**Uploaded:** {st.session_state.trade_name}")

save_name_mc = st.sidebar.text_input(
    "Save as (no .csv)", key="save_name_mc"
)
if st.sidebar.button("Save uploaded log", key="save_button_mc"):
    if not st.session_state.trade_bytes:
        st.sidebar.warning("No file to save.")
    elif not save_name_mc.strip():
        st.sidebar.warning("Please enter a name.")
    else:
        path = os.path.join(DATA_DIR, f"{save_name_mc}.csv")
        with open(path, "wb") as f:
            f.write(st.session_state.trade_bytes)
        st.sidebar.success(f"Saved as {save_name_mc}.csv")

st.sidebar.subheader("Saved Logs")
saved_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
selected_saved = st.sidebar.selectbox(
    "Load saved log", [""] + saved_files, key="selected_saved_file_mc"
)
if selected_saved and st.sidebar.button("Delete selected log", key="delete_button_mc"):
    os.remove(os.path.join(DATA_DIR, selected_saved))
    st.sidebar.success(f"Deleted {selected_saved}")
    st.experimental_rerun()

if selected_saved:
    uploaded = os.path.join(DATA_DIR, selected_saved)
elif st.session_state.trade_bytes:
    uploaded = io.BytesIO(st.session_state.trade_bytes)
else:
    uploaded = None

if not uploaded:
    st.info("Please upload a CSV with your P/L data (e.g. columns 'Date Opened' and 'P/L').")
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# Detect TS by peeking at header
if hasattr(uploaded, "getvalue"):
    first = uploaded.getvalue().splitlines()[0].decode("utf-8", "ignore")
else:
    with open(uploaded, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()

if "FinalTradeClosedDate" in first and "TotalNetProfitLoss" in first:
    # a TradeSteward log
    df = load_trade_steward(uploaded)
else:
    # Option Omega log
    if hasattr(uploaded, "seek"):
        uploaded.seek(0)
    try:
        df = pd.read_csv(uploaded, parse_dates=["Date Closed"])
    except pd.errors.EmptyDataError:
        st.error("⚠️ The selected CSV is empty or invalid.")
        st.stop()

# ────────────────────────────────────────────────────────────────────
# 🔄 Auto-detect OO vs. TS & unify into df["Date Closed"] / df["P/L"]

# 1) Option Omega exact headers
oo_date = next((c for c in df.columns if c.strip().lower()=="date closed"), None)
oo_pl   = next((c for c in df.columns if c.strip().lower()=="p/l"),          None)

# 2) TradeSteward headers by substring
ts_date = next(
    (c for c in df.columns
     if "finaltradecloseddate" in c.replace("_","").lower()),
    None
)
ts_pl   = next(
    (c for c in df.columns
     if "totalnetprofitloss"   in c.replace("_","").lower()),
    None
)

if oo_date and oo_pl:
    # Option Omega
    date_col, pl_col = oo_date, oo_pl
    df["Date Closed"] = pd.to_datetime(
        df[date_col], errors="coerce"
    )
    df["P/L"]         = pd.to_numeric(
        df[pl_col], errors="coerce"
    ).fillna(0)

elif ts_date and ts_pl:
    # TradeSteward: parse only the DATE (M/D/YYYY)
    date_col, pl_col = ts_date, ts_pl
    df["Date Closed"] = pd.to_datetime(
        df[date_col],
        format="%m/%d/%Y",
        errors="coerce"
    )
    df["P/L"]         = pd.to_numeric(
        df[pl_col],
        errors="coerce"
    ).fillna(0)

else:
    # Nothing auto-detected—let the user type in whatever
    date_col = st.text_input("Date column name", value="", key="auto_date_col")
    pl_col   = st.text_input("P/L column name",   value="", key="auto_pl_col")
    if not date_col or not pl_col:
        st.error("Please specify your date and P/L columns.")
        st.stop()
    df["Date Closed"] = pd.to_datetime(
        df[date_col], errors="coerce"
    )
    df["P/L"]         = pd.to_numeric(
        df[pl_col], errors="coerce"
    ).fillna(0)

# 3) Drop any rows where the date failed to parse
df = df.dropna(subset=["Date Closed"])

# 4) If that leaves us empty, bail out now
if df.empty:
    st.warning("⚠️ No valid trades after parsing your date/P&L columns.")
    st.stop()
# ────────────────────────────────────────────────────────────────────

st.title("Trading Performance Metrics")

# Aggregate P/L by close date and fill in any missing days
daily = df.groupby("Date Closed")["P/L"].sum().sort_index()
days  = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
daily = daily.reindex(days, fill_value=0).to_frame(name="P/L")

# ─── Date Range Presets (relative to latest trade date) ────────────────────
latest_date   = daily.index.max()
earliest_date = daily.index.min()

preset = st.selectbox("Date Range Presets", [
    "Today", "Yesterday", "Last 7 Days", "Last 30 Days", "Last 90 Days",
    "Last Month", "Week to Date", "Month to Date", "Year to Date",
    "Last Year", "Trailing 1-Year", "All Time", "Custom Range"
], index=11)  # default = "All Time"

if preset == "Today":
    start_date = end_date = latest_date

elif preset == "Yesterday":
    start_date = end_date = latest_date - datetime.timedelta(days=1)

elif preset == "Last 7 Days":
    start_date = latest_date - datetime.timedelta(days=6)
    end_date   = latest_date

elif preset == "Last 30 Days":
    start_date = latest_date - datetime.timedelta(days=29)
    end_date   = latest_date

elif preset == "Last 90 Days":
    start_date = latest_date - datetime.timedelta(days=89)
    end_date   = latest_date

elif preset == "Last Month":
    prev_month_end = latest_date.replace(day=1) - datetime.timedelta(days=1)
    start_date     = prev_month_end.replace(day=1)
    end_date       = prev_month_end

elif preset == "Week to Date":
    start_date = latest_date - datetime.timedelta(days=latest_date.weekday())
    end_date   = latest_date

elif preset == "Month to Date":
    start_date = latest_date.replace(day=1)
    end_date   = latest_date

elif preset == "Year to Date":
    start_date = latest_date.replace(month=1, day=1)
    end_date   = latest_date

elif preset == "Last Year":
    prev_year_end = latest_date.replace(month=1, day=1) - datetime.timedelta(days=1)
    start_date    = prev_year_end.replace(month=1, day=1)
    end_date      = prev_year_end

elif preset == "Trailing 1-Year":
    start_date = latest_date - relativedelta(years=1)
    end_date   = latest_date

elif preset == "All Time":
    start_date = earliest_date
    end_date   = latest_date

else:  # Custom Range
    custom = st.date_input(
        "Custom Date Range",
        value=(earliest_date, latest_date),
        min_value=earliest_date,
        max_value=latest_date,
    )
    # unpack safely even if user only clicks one date
    if isinstance(custom, (list, tuple)):
        if len(custom) >= 2:
            start_date, end_date = custom[0], custom[1]
        elif len(custom) == 1:
            start_date = end_date = custom[0]
        else:
            start_date = earliest_date
            end_date = latest_date
    else:
        start_date = end_date = custom

# show which slice we’re looking at
st.markdown(f"**Viewing:** {start_date.strftime('%m/%d/%Y')} – {end_date.strftime('%m/%d/%Y')}")

# convert to Timestamps so dtype matches
start_ts = pd.to_datetime(start_date)
end_ts   = pd.to_datetime(end_date)
mask     = (daily.index >= start_ts) & (daily.index <= end_ts)
daily    = daily.loc[mask]

if daily.empty:
    st.warning("⚠️ No trades found in that date range.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────

# now rebuild your returns & count
daily_returns = daily.values
N             = len(daily_returns)
if N == 0:
    st.warning("⚠️ No P/L data after aggregating. Check your date/P&L column mapping.")
    st.stop()

# Compute equity and drawdown
daily["Equity"]      = daily["P/L"].cumsum()
daily["RunningPeak"] = daily["Equity"].cummax()
daily["Drawdown"]    = daily["RunningPeak"] - daily["Equity"]

# ─── Optional Starting Capital ─────────────────────────────────────────────────
starting_cap_input = st.number_input(
    "Starting Capital (optional)",
    min_value=0.0,
    value=0.0,
    step=100.0,
    format="%.2f",
)
starting_capital = starting_cap_input if starting_cap_input > 0 else None

# ─── 1) Win/Loss counts & pct ──────────────────────────────────────────────────
wins        = df["P/L"] > 0
losses      = df["P/L"] < 0
win_count   = int(wins.sum())
loss_count  = int(losses.sum())
total_trades = win_count + loss_count
win_pct     = win_count / max(total_trades, 1) * 100.0

# ─── 2) Daily equity & drawdown ────────────────────────────────────────────────
cum_equity  = daily["Equity"]
daily_dd    = daily["Drawdown"]
current_dd  = daily_dd.iloc[-1]
max_dd      = daily_dd.max()

# ─── 3) Days in current drawdown ───────────────────────────────────────────────
last_ath    = cum_equity.cummax().values.argmax()
days_in_dd  = len(daily) - last_ath - 1

# ─── 4) Profit run since last recovery ─────────────────────────────────────────
dd_zero     = daily_dd == 0
if dd_zero.any():
    last_reset   = daily.index[dd_zero][-1]
else:
    last_reset   = daily.index[0]
profit_run  = cum_equity.iloc[-1] - cum_equity.loc[last_reset]

# split into mutually‐exclusive profit vs drawdown
net_profit       = daily['P/L'].sum()
current_drawdown = max(-min(profit_run, 0.0), 0.0)

# ─── 5) Current streak (days) ─────────────────────────────────────────────────────
pnl_series = daily["P/L"]

# Determine streak type by last non-zero day
nonzero = pnl_series[pnl_series != 0]
if not nonzero.empty:
    last_val = nonzero.iloc[-1]
    if last_val > 0:
        current_streak_type = "Win"
        predicate = lambda x: x >= 0
    else:
        current_streak_type = "Loss"
        predicate = lambda x: x <= 0

    # Count consecutive days (including zeros) matching that predicate
    current_streak_count = 0
    for x in pnl_series.iloc[::-1]:
        if predicate(x):
            current_streak_count += 1
        else:
            break

    # Singular vs. plural
    if current_streak_type == "Win":
        # Win → Wins
        suffix = "s" if current_streak_count > 1 else ""
    elif current_streak_type == "Loss":
        # Loss → Losses
        suffix = "es" if current_streak_count > 1 else ""
    else:
        suffix = ""

    if current_streak_count > 0 and current_streak_type:
        streak_label = f"{current_streak_count} {current_streak_type}{suffix}"
    else:
        streak_label = "0"
else:
    streak_label = "0"

# ─── 6) Averages ───────────────────────────────────────────────────────────────
avg_daily_win   = daily.loc[daily["P/L"] > 0, "P/L"].mean()   or 0.0
avg_daily_loss  = daily.loc[daily["P/L"] < 0, "P/L"].mean()   or 0.0
avg_trade_win   = df.loc[wins,   "P/L"].mean()               or 0.0
avg_trade_loss  = df.loc[losses, "P/L"].mean()               or 0.0

# ─── 7) Extremes ───────────────────────────────────────────────────────────────
biggest_trade_win  = df["P/L"].max()
biggest_trade_loss = df["P/L"].min()
biggest_day_win    = daily["P/L"].max()
biggest_day_loss   = daily["P/L"].min()

# ─── 8) Total profit & CAGR ───────────────────────────────────────────────────
total_profit = daily["P/L"].sum()

if starting_capital is not None:
    end_equity = starting_capital + total_profit
    days_diff  = (daily.index.max() - daily.index.min()).days
    if days_diff > 0:
        years = days_diff / 365.25
        cagr  = (end_equity / starting_capital) ** (1 / years) - 1
        cagr *= 100.0
    else:
        # single day or zero-length range → no annualization
        cagr = 0.0
else:
    days_count = (daily.index.max() - daily.index.min()).days + 1
    if days_count > 1:
        cagr = ((cum_equity.iloc[-1] / cum_equity.iloc[0]) ** (252 / days_count) - 1) * 100.0
    else:
        # single-day fallback
        cagr = 0.0

# ─── Daily Win/Loss Streaks (calendar days) ────────────────────────────────────
win_runs  = []
loss_runs = []
temp_run  = 0

# Winning‐day runs: count days where P/L ≥ 0
for pnl in daily["P/L"]:
    if pnl >= 0:
        temp_run += 1
    else:
        if temp_run > 0:
            win_runs.append(temp_run)
        temp_run = 0
# catch final
if temp_run > 0:
    win_runs.append(temp_run)

# Losing‐day runs: count days where P/L ≤ 0
temp_run = 0
for pnl in daily["P/L"]:
    if pnl <= 0:
        temp_run += 1
    else:
        if temp_run > 0:
            loss_runs.append(temp_run)
        temp_run = 0
if temp_run > 0:
    loss_runs.append(temp_run)

max_win_streak  = max(win_runs)  if win_runs  else 0
max_loss_streak = max(loss_runs) if loss_runs else 0

# ─── 9) Flatten into lookup dict ────────────────────────────────────────────────
metrics = {
    # Snapshot
    "Net Profit ($)":        f"{net_profit:,.2f}",
    "Current Streak (Days)": streak_label,
    "Days in Drawdown":      f"{days_in_dd} days",
    "Current Drawdown ($)":  f"{current_drawdown:,.2f}",

    # Core & trade‐level
    "CAGR (%)":              f"{cagr:.1f}%",
    "Trade Count":           f"{total_trades}",
    "Win %":                 f"{win_pct:.1f}%",

    # Averages
    "Avg Daily Win":         f"{avg_daily_win:,.2f}",
    "Avg Daily Loss":        f"{avg_daily_loss:,.2f}",
    "Avg Trade Win":         f"{avg_trade_win:.2f}",
    "Avg Trade Loss":        f"{avg_trade_loss:.2f}",

    # Streaks & drawdown
    "Longest Win Streak":    f"{max_win_streak} days",
    "Longest Loss Streak":   f"{max_loss_streak} days",
    "Max Drawdown ($)":      f"{max_dd:,.2f}",

    # Extremes
    "Biggest Trade Win":     f"{biggest_trade_win:,.2f}",
    "Biggest Trade Loss":    f"{biggest_trade_loss:,.2f}",
    "Biggest Day Win":       f"{biggest_day_win:,.2f}",
    "Biggest Day Loss":      f"{biggest_day_loss:,.2f}",
}

# ─── Snapshot Metrics Panel ────────────────────────────────────────────────────
st.header("Snapshot Metrics")
snapshot_keys = [
    "Net Profit ($)",
    "Current Streak (Days)",
    "Days in Drawdown",
    "Current Drawdown ($)",
]
for i in range(0, len(snapshot_keys), 4):
    row = snapshot_keys[i : i + 4]
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        col.metric(name, metrics[name])

# ─── Key Metrics (selectable) ──────────────────────────────────────────────────
st.header("Key Metrics")
# Exclude snapshot items
key_metrics_map = {k: v for k, v in metrics.items() if k not in snapshot_keys}
default_key_metrics = list(key_metrics_map.keys())

if st.button("Reset Key Metrics"):
    st.session_state.pop("key_metrics", None)

chosen_keys = st.multiselect(
    "Select Key Metrics to display",
    options=default_key_metrics,
    default=default_key_metrics,
    key="key_metrics",
)

if chosen_keys:
    for i in range(0, len(chosen_keys), 4):
        row = chosen_keys[i : i + 4]
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            col.metric(name, key_metrics_map[name])
else:
    st.info("No Key Metrics selected. Use the dropdown above to pick metrics.")

# ─── ( next up: your equity-curve plot ) ───────────────────────────────────────

# Plot equity curve and drawdown
fig, ax = plt.subplots(figsize=(10, 5))  # make it a bit less gigantic
# convert our datetime‐index into Matplotlib floats
dt_idx      = pd.to_datetime(daily.index)
x_vals      = mdates.date2num(dt_idx.to_pydatetime())
equity_vals = daily["Equity"].values
peak_vals   = daily["RunningPeak"].values
dd_mask     = daily["Drawdown"].values > 0

ax.plot(x_vals, equity_vals, "-", label="Equity Curve")
ax.fill_between(
    x_vals,
    equity_vals,
    peak_vals,
    where=dd_mask,
    alpha=0.3,
    label="Drawdown"
)

# smarter ticks & rotated labels
locator = mdates.AutoDateLocator(maxticks=15)
formatter = mdates.AutoDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
fig.autofmt_xdate()

ax.set_xlabel("Date")
ax.set_ylabel("Equity")
ax.legend()
st.pyplot(fig)

# Show raw daily data
st.header("Daily P/L Data")
st.dataframe(daily[["P/L", "Equity", "Drawdown"]])
