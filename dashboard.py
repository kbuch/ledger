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
st.title("📊 Dashboard: Trading Performance")

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

st.title("Trading Performance Dashboard")

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

# Compute streaks and metrics
def compute_streaks(series):
    return [len(list(group)) for key, group in groupby(series) if key]

daily["Win"]   = daily["P/L"] >  0
daily["Loss"]  = daily["P/L"] <  0

win_streaks   = compute_streaks(daily["Win"])
loss_streaks  = compute_streaks(daily["Loss"])

avg_win_streak      = sum(win_streaks)  / len(win_streaks)  if win_streaks  else 0
avg_loss_streak     = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
longest_win_streak  = max(win_streaks)  if win_streaks  else 0
longest_loss_streak = max(loss_streaks) if loss_streaks else 0
avg_win             = daily.loc[daily["Win"],  "P/L"].mean() if daily["Win"].any()  else 0
avg_loss            = daily.loc[daily["Loss"], "P/L"].mean() if daily["Loss"].any() else 0

max_drawdown      = daily["Drawdown"].max()
current_drawdown  = daily["Drawdown"].iloc[-1]
last_ath_idx      = daily["Equity"].cummax().values.argmax()
days_in_dd        = len(daily) - last_ath_idx - 1

# Display metrics
st.header("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Win Streak",      f"{avg_win_streak:.2f} days")
col1.metric("Avg Loss Streak",     f"{avg_loss_streak:.2f} days")
col2.metric("Longest Win Streak",  f"{longest_win_streak} days")
col2.metric("Longest Loss Streak", f"{longest_loss_streak} days")
col3.metric("Avg Win",             f"${avg_win:.2f}")
col3.metric("Avg Loss",            f"${avg_loss:.2f}")
col1.metric("Max Drawdown",        f"${max_drawdown:.2f}")
col2.metric("Current Drawdown",    f"${current_drawdown:.2f}")
col3.metric("Days in Drawdown",    f"{days_in_dd}")

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
st.subheader("Daily P/L Data")
st.dataframe(daily[["P/L", "Equity", "Drawdown"]])
