# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
import os
import io

# ──────────────────────────────────────────────────────────────────────
# 1) Ensure data directory exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 2) Initialize shared session_state slots
if "trade_bytes" not in st.session_state:
    st.session_state.trade_bytes = None
    st.session_state.trade_name  = None

# 3) Sidebar: Data Manager
st.sidebar.header("📁 Data Manager")

#    a) upload new (same key on both pages!)
uploaded_file = st.sidebar.file_uploader(
    "Upload trade-log CSV", type="csv", key="trade_uploader"
)
if uploaded_file is not None:
    # stash raw bytes + filename
    st.session_state.trade_bytes = uploaded_file.getvalue()
    st.session_state.trade_name  = uploaded_file.name

#    b) show current upload
if st.session_state.trade_name:
    st.sidebar.markdown(f"**Uploaded:** {st.session_state.trade_name}")

#    c) save uploaded to disk
save_name = st.sidebar.text_input("Save as (no .csv)", key="save_name")
if st.sidebar.button("Save uploaded log"):
    if not st.session_state.trade_bytes:
        st.sidebar.warning("No file to save.")
    elif not save_name.strip():
        st.sidebar.warning("Please enter a name.")
    else:
        path = os.path.join(DATA_DIR, f"{save_name}.csv")
        with open(path, "wb") as f:
            f.write(st.session_state.trade_bytes)
        st.sidebar.success(f"Saved as {save_name}.csv")

#    d) list / delete persisted CSVs
st.sidebar.subheader("Saved Logs")
saved_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
selected_saved = st.sidebar.selectbox(
    "Load saved log", [""] + saved_files, key="selected_saved_file"
)
if selected_saved and st.sidebar.button("Delete selected log"):
    os.remove(os.path.join(DATA_DIR, selected_saved))
    st.sidebar.success(f"Deleted {selected_saved}")
    st.experimental_rerun()

# 4) Determine which CSV to load
if selected_saved:
    df = pd.read_csv(
        os.path.join(DATA_DIR, selected_saved),
        parse_dates=["Date Closed"],
    )
elif st.session_state.trade_bytes:
    df = pd.read_csv(
        io.BytesIO(st.session_state.trade_bytes),
        parse_dates=["Date Closed"],
    )
else:
    st.info("Please upload or select a CSV trade-log to begin.")
    st.stop()
# ──────────────────────────────────────────────────────────────────────

# ----- Your original dashboard code below, unchanged -----

st.title("Trading Performance Dashboard")

# Read and preprocess data
daily = df.groupby("Date Closed")["P/L"].sum().sort_index().reset_index()

# Determine full-range bounds
min_date = daily["Date Closed"].min()
max_date = daily["Date Closed"].max()
today    = pd.Timestamp.today().normalize()

# Preset range selector
preset = st.selectbox(
    "Date Range Presets",
    [
        "All Time",
        "Year to Date",
        "Last 30 Days",
        "Last 60 Days",
        "Last 90 Days",
        "Last Year",
        "Today Only",
        "Custom",
    ],
    index=0,
)

# Compute start/end based on preset
if preset == "All Time":
    start_date, end_date = min_date, max_date
elif preset == "Year to Date":
    start_date = pd.Timestamp(year=today.year, month=1, day=1)
    end_date   = today if today <= max_date else max_date
elif preset == "Last 30 Days":
    start_date = max(today - pd.Timedelta(days=30), min_date)
    end_date   = today  if today <= max_date else max_date
elif preset == "Last 60 Days":
    start_date = max(today - pd.Timedelta(days=60), min_date)
    end_date   = today  if today <= max_date else max_date
elif preset == "Last 90 Days":
    start_date = max(today - pd.Timedelta(days=90), min_date)
    end_date   = today  if today <= max_date else max_date
elif preset == "Last Year":
    prev_year  = today.year - 1
    start_date = pd.Timestamp(year=prev_year, month=1,  day=1)
    end_date   = pd.Timestamp(year=prev_year, month=12, day=31)
    start_date = max(start_date, min_date)
    end_date   = min(end_date,   max_date)
elif preset == "Today Only":
    if today < min_date:
        start_date = end_date = min_date
    elif today > max_date:
        start_date = end_date = max_date
    else:
        start_date = end_date = today
else:  # Custom
    start_date, end_date = st.date_input(
        "Select start and end dates",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if start_date > end_date:
        st.error("❗ Start date must be before end date.")
        st.stop()

# Filter daily series
mask = (
    (daily["Date Closed"] >= pd.to_datetime(start_date)) &
    (daily["Date Closed"] <= pd.to_datetime(end_date))
)
daily = daily.loc[mask].reset_index(drop=True)

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
last_ath_idx      = daily["Equity"].cummax().idxmax()
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
fig, ax = plt.subplots()
ax.plot(daily["Date Closed"], daily["Equity"], label="Equity Curve")
ax.fill_between(
    daily["Date Closed"],
    daily["Equity"],
    daily["RunningPeak"],
    where=daily["Drawdown"] > 0,
    alpha=0.3,
    label="Drawdown"
)
ax.set_xlabel("Date")
ax.set_ylabel("Equity")
ax.legend()
st.pyplot(fig)

# Show raw daily data
st.subheader("Daily P/L Data")
st.dataframe(
    daily.set_index("Date Closed")[["P/L", "Equity", "Drawdown"]]
)
