# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
import os
import io

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

if daily.empty:
    st.warning("⚠️ No trades found in that date range. "
               "Please check your date column mapping or adjust the range.")
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
