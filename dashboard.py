# dashboard.py

import io
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from scipy import stats

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

# ─── Reset / Refresh App ──────────────────────────────────────────────────────
if st.sidebar.button("🧹 Reset App"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
        # restart the script
    st.rerun()

if st.sidebar.button("🔄 Refresh App"):
    st.session_state.pop("key_metrics", None)

# ─── Data Manager: shared upload + save/load/delete ───────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# initialize once
if "trade_bytes" not in st.session_state:
    st.session_state.trade_bytes = None
    st.session_state.trade_name  = None

st.sidebar.header("📁 Data Manager")
uploader = st.sidebar.file_uploader(
    "Upload trade-log CSV", type="csv", key="trade_uploader"
)
if uploader is not None:
    st.session_state.trade_bytes = uploader.getvalue()
    st.session_state.trade_name  = uploader.name

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

# ←――――――――――――――――――――――――――――
#  CAPTURE RAW IMPORT FOR VALIDATION
raw_import = df[["Date Closed", "P/L"]].copy()

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

# ←――――――――――――――――――――――――――――
#  CAPTURE CLEANED IMPORT FOR VALIDATION
cleaned_import = df[["Date Closed", "P/L"]].copy()

# ────────────────────────────────────────────────────────────────────

st.title("Trading Performance Metrics")

# ─── Determine full data bounds for presets ───────────────────────────────────
earliest_date = df["Date Closed"].dt.normalize().min()
latest_date   = df["Date Closed"].dt.normalize().max()

# ─── Date Range Presets (relative to latest trade date) ────────────────────
preset = st.selectbox("Date Range Presets", [
    "Today", "Yesterday", "Last 7 Days", "Last 30 Days", "Last 90 Days",
    "Last Month", "Week to Date", "Month to Date", "Year to Date",
    "Last Year", "Trailing 1-Year", "All Time", "SPX Dailies", "Custom Range"
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

elif preset == "SPX Dailies":
    start_date = pd.to_datetime("2022-05-16")
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

# convert to normalized Timestamps
start_ts = pd.to_datetime(start_date).normalize()
end_ts   = pd.to_datetime(end_date).normalize()

# ─── 1) Slice the raw trades DataFrame to your window ────────────────────────
df = df.loc[
    (df["Date Closed"].dt.normalize() >= start_ts) &
    (df["Date Closed"].dt.normalize() <= end_ts)
]
if df.empty:
    st.warning("⚠️ No trades found in that date range.")
    st.stop()

# ─── 2) Build the daily P/L series over *every* calendar day ────────────────
#   a) Aggregate P/L on actual trade-dates
daily_raw = (df.groupby(df["Date Closed"].dt.normalize())["P/L"].sum().sort_index())
#   b) Create full calendar index for the exact window
calendar = pd.date_range(start=start_ts, end=end_ts, freq="D")
#   c) Zero-fill days with no trades
daily = daily_raw.reindex(calendar, fill_value=0).to_frame(name="P/L")

# ─── DEBUG / DATA VALIDATION (Dev Only) ─────────────────────────────────────
raw_import     = raw_import.copy()
cleaned_import = cleaned_import.copy()

if st.checkbox("🔍 Show data validation (dev only)", key="show_data_debug"):
    with st.expander("⚠️ Data Validation"):
        # 1) Pre-slice sanity:
        st.write("• Rows at initial load:",           len(raw_import))
        st.write("• Rows after cleaning:",            len(cleaned_import))
        st.write("• Total P/L at initial load:",      f"{raw_import['P/L'].sum():,.2f}")
        st.write("• Total P/L after cleaning:",       f"{cleaned_import['P/L'].sum():,.2f}")

        # 2) Date-window masks
        mask_raw   = (raw_import["Date Closed"]   >= start_ts) & (raw_import["Date Closed"]   <= end_ts)
        mask_clean = (cleaned_import["Date Closed"] >= start_ts) & (cleaned_import["Date Closed"] <= end_ts)

        raw_window    = raw_import.loc[mask_raw]
        clean_window  = cleaned_import.loc[mask_clean]

        st.write("• Rows in date range (raw):",     len(raw_window))
        st.write("• Rows in date range (clean):",   len(clean_window))
        st.write("• Rows after slicing on df:",      len(df))

        # 3) Show dropped rows, if any
        if len(raw_window) != len(df):
            dropped_idx  = raw_window.index.difference(df.index)
            dropped_rows = raw_import.loc[dropped_idx]
            st.write("❌ Trades in raw window that did NOT make it into df:")
            st.dataframe(dropped_rows)

        else:
            st.write("✅ No trades lost during final slice.")

# ─────────────────────────────────────────────────────────────────────────────

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
if starting_capital is not None:
    end_equity = starting_capital + net_profit
    days_diff  = (daily.index.max() - daily.index.min()).days
    # only annualize if we have a multi-day span and a positive start
    if days_diff > 0 and starting_capital > 0:
        years = days_diff / 365.25
        # suppress any invalid‐power or divide warnings here
        with np.errstate(invalid="ignore", divide="ignore"):
            cagr = (end_equity / starting_capital) ** (1 / years) - 1
        cagr *= 100.0
    else:
        cagr = 0.0
else:
    days_count = (daily.index.max() - daily.index.min()).days + 1
    first_eq   = cum_equity.iloc[0]
    last_eq    = cum_equity.iloc[-1]
    # only annualize if >1 day span and non-zero first equity
    if days_count > 1 and first_eq > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            cagr = ((last_eq / first_eq) ** (252 / days_count) - 1) * 100.0
    else:
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

# Compute additional metrics

# 1) Max Drawdown (%) relative to starting capital (or “–” if none provided)
if starting_capital is not None:
    max_dd_pct = max_dd / starting_capital * 100.0
else:
    max_dd_pct = None

# 3) Average daily and per-trade P/L
avg_daily_pnl = daily["P/L"].mean() if not daily["P/L"].empty else 0.0
avg_trade_pnl = df["P/L"].mean()  if not df["P/L"].empty   else 0.0

# 4) Average win/loss streak lengths (in days)
avg_win_streak  = sum(win_runs)  / len(win_runs)  if win_runs  else 0.0
avg_loss_streak = sum(loss_runs) / len(loss_runs) if loss_runs else 0.0

# 5) Period ROI % over the selected date range
if starting_capital is not None and starting_capital > 0:
    end_equity      = starting_capital + net_profit
    period_roi_pct  = (end_equity / starting_capital - 1) * 100.0
elif cum_equity.iloc[0] != 0:
    period_roi_pct  = (cum_equity.iloc[-1] / cum_equity.iloc[0] - 1) * 100.0
else:
    period_roi_pct  = None

# ─── 8.5) Additional Risk & Reward Metrics ────────────────────────────────────

# Profit Factor (Option Alpha style)
gross_profit    = df.loc[df["P/L"] > 0, "P/L"].sum()
gross_loss      = -df.loc[df["P/L"] < 0, "P/L"].sum()
profit_factor   = gross_profit / gross_loss if gross_loss > 0 else None

# Reward:Risk = avg trade win / abs(avg trade loss)
reward_risk     = avg_trade_win / abs(avg_trade_loss) if avg_trade_loss != 0 else None

# Biggest Win/Loss Streak ($)  → the $.sum() over the same calendar-day streaks
win_streak_vals  = []
loss_streak_vals = []
_temp_vals      = []
_temp_type      = None

for pnl in daily["P/L"]:
    kind = "win" if pnl >= 0 else "loss"
    if _temp_type is None or kind == _temp_type:
        _temp_vals.append(pnl)
        _temp_type = kind
    else:
        if _temp_type == "win":
            win_streak_vals.append(sum(_temp_vals))
        else:
            loss_streak_vals.append(sum(_temp_vals))
        _temp_vals = [pnl]
        _temp_type = kind
# flush last run
if _temp_vals:
    if _temp_type == "win":
        win_streak_vals.append(sum(_temp_vals))
    else:
        loss_streak_vals.append(sum(_temp_vals))

biggest_win_streak_pnl  = max(win_streak_vals)  if win_streak_vals  else 0.0
biggest_loss_streak_pnl = min(loss_streak_vals) if loss_streak_vals else 0.0

# MAR Ratio = CAGR% / Max Drawdown% (both in absolute terms)
mar_ratio = (cagr or 0) / (max_dd_pct or 1) if max_dd_pct and max_dd_pct > 0 else None

# ─── Sharpe & Sortino Ratios ─────────────────────────────────────────
returns = daily["Equity"].pct_change().dropna()

if len(returns) > 1:
    mu = returns.mean()

    # population‐std or sample‐std is up to you; ddof=0 uses population
    sigma = returns.std(ddof=0)
    downside = returns[returns < 0]
    downside_sigma = downside.std(ddof=0) if len(downside) > 0 else 0.0

    sharpe_ratio = (mu / sigma * np.sqrt(252)) if sigma > 0 else None
    sortino_ratio = (mu / downside_sigma * np.sqrt(252)) if downside_sigma > 0 else None
else:
    sharpe_ratio = None
    sortino_ratio = None

# ─── New “Advanced” Metrics Calculations ────────────────────────────────────

# 1) Expectancy ($ per trade)
p_win   = win_pct / 100.0
p_loss  = 1 - p_win
expectancy = p_win * avg_trade_win + p_loss * avg_trade_loss

# 2) Kelly Fraction
# Kelly = p – q / R  where R = avg_trade_win/abs(avg_trade_loss)
kelly_fraction = None
if reward_risk and reward_risk > 0:
    kf = p_win - (p_loss / reward_risk)
    kelly_fraction = kf

# 3) Time in Drawdown (% of calendar days below peak)
time_in_drawdown = daily["Drawdown"].gt(0).sum() / len(daily) * 100.0

# 4) Annualized Volatility (%)
daily_ret  = daily["Equity"].pct_change().dropna()
annual_vol = daily_ret.std(ddof=0) * np.sqrt(252) * 100.0 if not daily_ret.empty else None

# 5) Ulcer Index (%)
# drawdown percent series
dd_pct     = daily["Drawdown"] / daily["RunningPeak"].replace(0, np.nan)
ulcer_index = np.sqrt((dd_pct.dropna() ** 2).mean()) * 100.0 if not dd_pct.dropna().empty else None

# 6) Average Holding Period (days) — requires df["Date Opened"]
average_holding = None
if "Date Opened" in df.columns:
    avg_hold = (df["Date Closed"] - pd.to_datetime(df["Date Opened"])).dt.days
    average_holding = avg_hold.mean()

# 7) R-Squared of Equity Trend
if len(daily) > 1:
    x = np.arange(len(daily))
    y = daily["Equity"].values
    # Pearson r
    r = np.corrcoef(x, y)[0,1]
    r_squared = r**2
else:
    r_squared = None

# ─── 1) Define every metric in one place ────────────────────────────────────
metric_definitions = [
    {
        "key":         "Net Profit ($)",
        "value_fn":    lambda: net_profit,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "The total dollars gained or lost over the period—your absolute P/L.",
    },
    {
        "key":         "Period ROI (%)",
        "value_fn":    lambda: period_roi_pct,
        "fmt":         lambda x: f"{x:.1f}%" if x is not None else "-",
        "description": "Your total return expressed as a percentage of starting equity; shows overall efficiency.",
    },
    {
        "key":         "Current Streak (Days)",
        "value_fn":    lambda: streak_label,
        "fmt":         lambda x: x,
        "description": "How many days in a row you’ve been profitable (or unprofitable); momentum indicator.",
    },
    {
        "key":         "Days in Drawdown",
        "value_fn":    lambda: days_in_dd,
        "fmt":         lambda x: f"{x} days",
        "description": "How long it’s been since you hit a new high—measures recovery time under stress.",
    },
    {
        "key":         "Current Drawdown ($)",
        "value_fn":    lambda: current_drawdown,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "The dollar gap between your peak equity and today—gauges your maximum recent loss.",
    },
    {
        "key":         "CAGR (%)",
        "value_fn":    lambda: cagr,
        "fmt":         lambda x: f"{x:.1f}%",
        "description": "Compound annual growth rate—normalizes return to a yearly basis for easy comparison.",
    },
    {
        "key":         "Trade Count",
        "value_fn":    lambda: total_trades,
        "fmt":         lambda x: str(x),
        "description": "Total number of executions—shows activity level and sample size.",
    },
    {
        "key":         "Wins",
        "value_fn":    lambda: win_count,
        "fmt":         lambda x: str(x),
        "description": "Count of profitable trades—raw tally of positive outcomes.",
    },
    {
        "key":         "Losses",
        "value_fn":    lambda: loss_count,
        "fmt":         lambda x: str(x),
        "description": "Count of unprofitable trades—raw tally of negative outcomes.",
    },
    {
        "key":         "Win Rate (%)",
        "value_fn":    lambda: win_pct,
        "fmt":         lambda x: f"{x:.1f}%",
        "description": "Proportion of trades that won—basic measure of your edge.",
    },
    {
        "key":         "Avg Daily Win ($)",
        "value_fn":    lambda: avg_daily_win,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Mean profit on winning days—shows typical upside per profitable session.",
    },
    {
        "key":         "Avg Daily Loss ($)",
        "value_fn":    lambda: avg_daily_loss,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Mean loss on losing days—shows typical downside per losing session.",
    },
    {
        "key":         "Avg Daily P/L ($)",
        "value_fn":    lambda: avg_daily_pnl,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Average net P/L per calendar day, including days with no trades—measures consistency.",
    },
    {
        "key":         "Avg Trade Win ($)",
        "value_fn":    lambda: avg_trade_win,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Average profit on winning trades—assesses reward per successful trade.",
    },
    {
        "key":         "Avg Trade Loss ($)",
        "value_fn":    lambda: avg_trade_loss,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Average loss on losing trades—assesses risk per unsuccessful trade.",
    },
    {
        "key":         "Avg Trade P/L ($)",
        "value_fn":    lambda: avg_trade_pnl,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Net average P/L per trade—combines win/loss magnitude and frequency.",
    },
    {
        "key":         "Longest Win Streak (Days)",
        "value_fn":    lambda: max_win_streak,
        "fmt":         lambda x: f"{x} days",
        "description": "Longest run of daily non-negative P/L—indicates sustained performance periods.",
    },
    {
        "key":         "Longest Loss Streak (Days)",
        "value_fn":    lambda: max_loss_streak,
        "fmt":         lambda x: f"{x} days",
        "description": "Longest run of daily non-positive P/L—highlights toughest drawdown periods.",
    },
    {
        "key":         "Avg Win Streak (Days)",
        "value_fn":    lambda: avg_win_streak,
        "fmt":         lambda x: f"{x:.1f} days",
        "description": "Average length of winning‐day runs—measures typical momentum clusters.",
    },
    {
        "key":         "Avg Loss Streak (Days)",
        "value_fn":    lambda: avg_loss_streak,
        "fmt":         lambda x: f"{x:.1f} days",
        "description": "Average length of losing‐day runs—measures typical downturn clusters.",
    },
    {
        "key":         "Max Drawdown ($)",
        "value_fn":    lambda: max_dd,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Largest dollar equity drop from peak—key measure of worst-case risk.",
    },
    {
        "key":         "Max Drawdown (%)",
        "value_fn":    lambda: max_dd_pct,
        "fmt":         lambda x: f"{x:.1f}%" if x is not None else "-",
        "description": "Largest percentage equity drop from peak—standardized risk metric.",
    },
    {
        "key":         "Biggest Trade Win",
        "value_fn":    lambda: biggest_trade_win,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "The single largest P/L from one trade—shows your maximum upside potential.",
    },
    {
        "key":         "Biggest Trade Loss",
        "value_fn":    lambda: biggest_trade_loss,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "The single largest loss from one trade—shows your maximum downside hit.",
    },
    {
        "key":         "Biggest Day Win",
        "value_fn":    lambda: biggest_day_win,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Single day with the highest net P/L—identifies standout performance days.",
    },
    {
        "key":         "Biggest Day Loss",
        "value_fn":    lambda: biggest_day_loss,
        "fmt":         lambda x: f"{x:,.2f}",
        "description": "Single day with the largest net loss—highlights worst performance days.",
    },
    {
        "key":         "MAR Ratio",
        "value_fn":    lambda: mar_ratio,
        "fmt":         lambda x: f"{x:.2f}" if x is not None else "-",
        "description": "CAGR ÷ max drawdown—shows risk‐adjusted return efficiency.",
    },
    {
        "key":         "Profit Factor",
        "value_fn":    lambda: profit_factor,
        "fmt":         lambda x: f"{x:.2f}" if x is not None else "-",
        "description": "Gross profits ÷ gross losses—indicates overall trade profitability.",
    },
    {
        "key":         "Reward:Risk",
        "value_fn":    lambda: reward_risk,
        "fmt":         lambda x: f"{x:.2f}" if x is not None else "-",
        "description": "Average reward per $1 risked—measures efficiency of reward vs. risk.",
    },
    {
        "key":         "Sharpe Ratio",
        "value_fn":    lambda: sharpe_ratio,
        "fmt":         lambda x: f"{x:.2f}" if x is not None else "-",
        "description": "Excess return per unit of volatility—standard risk‐adjusted return metric.",
    },
    {
        "key":         "Sortino Ratio",
        "value_fn":    lambda: sortino_ratio,
        "fmt":         lambda x: f"{x:.2f}" if x is not None else "-",
        "description": "Excess return per unit of downside volatility—focuses on harmful swings.",
    },
    {
        "key": "Expectancy ($)",
        "value_fn": lambda: expectancy,
        "fmt": lambda x: f"{x:,.2f}",
        "description": "Average $ gained (or lost) per trade; measures your edge.",
    },
    {
        "key": "Kelly Fraction",
        "value_fn": lambda: kelly_fraction,
        "fmt": lambda x: f"{x:.2f}" if x is not None else "-",
        "description": "Optimal bet‐size fraction: p – q/R, where R is reward-to-risk.",
    },
    {
        "key": "Time in Drawdown (%)",
        "value_fn": lambda: time_in_drawdown,
        "fmt": lambda x: f"{x:.1f}%",
        "description": "Percentage of days spent below previous equity peaks.",
    },
    {
        "key": "Ann. Volatility (%)",
        "value_fn": lambda: annual_vol,
        "fmt": lambda x: f"{x:.1f}%" if x is not None else "-",
        "description": "Std dev of daily returns, annualized—measures curve ‘bumpiness.’",
    },
    {
        "key": "Ulcer Index (%)",
        "value_fn": lambda: ulcer_index,
        "fmt": lambda x: f"{x:.1f}%" if x is not None else "-",
        "description": "Drawdown depth+duration metric: √mean(%drawdown²).",
    },
    {
        "key": "Avg Holding Period (Days)",
        "value_fn": lambda: average_holding,
        "fmt": lambda x: f"{x:.1f} days" if x is not None else "-",
        "description": "Mean time between trade open and close.",
    },
    {
        "key": "Equity R²",
        "value_fn": lambda: r_squared,
        "fmt": lambda x: f"{x:.3f}" if x is not None else "-",
        "description": "R-squared of equity vs. time; higher ⇒ steadier trend.",
    },
]

# ─── 2) Build formatted metrics + description map ─────────────────────────────
metrics = {}
metric_desc_map = {}
for md in metric_definitions:
    key = md["key"]
    raw = md["value_fn"]()
    try:
        formatted = md["fmt"](raw) if raw is not None else "-"
    except Exception:
        formatted = "-"
    metrics[key] = formatted
    metric_desc_map[key] = md["description"]

# ─── 3) Snapshot Metrics Panel ───────────────────────────────────────────────
st.header("📸 Snapshot Metrics")
snapshot_keys = [md["key"] for md in metric_definitions[:5]]
cols = st.columns(len(snapshot_keys), gap="large")
for col, name in zip(cols, snapshot_keys):
    col.metric(label=name, value=metrics[name], help=metric_desc_map[name])

# ─── 4) Key Metrics Panel ───────────────────────────────────────────────────
st.header("🔑 Key Metrics")

with st.expander("⚙️ Configure Key Metrics", expanded=False):
    if st.button("Reset Key Metrics"):
        st.session_state.pop("key_metrics", None)

    available = [md["key"] for md in metric_definitions if md["key"] not in snapshot_keys]
    chosen = st.multiselect(
        "Pick metrics to display",
        options=available,
        default=st.session_state.get("key_metrics", available),
        key="key_metrics",
    )

if not chosen:
    st.info("Select at least one metric above.")
else:
    for i in range(0, len(chosen), 4):
        row = chosen[i : i + 4]
        cols = st.columns(len(row), gap="large")
        for col, name in zip(cols, row):
            col.metric(label=name, value=metrics[name], help=metric_desc_map[name])

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
