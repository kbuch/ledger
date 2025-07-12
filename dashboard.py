import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby

st.title("Trading Performance Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your trade log (CSV)", type="csv")
if uploaded_file is not None:
    # Read and preprocess data
    df = pd.read_csv(uploaded_file, parse_dates=["Date Closed"])
    daily = df.groupby("Date Closed")["P/L"].sum().sort_index().reset_index()

    # Determine full-range bounds
    min_date = daily["Date Closed"].min()
    max_date = daily["Date Closed"].max()
    today = pd.Timestamp.today().normalize()

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
            "Custom"
        ],
        index=0
    )

    # Compute start/end based on preset
    if preset == "All Time":
        start_date, end_date = min_date, max_date
    elif preset == "Year to Date":
        start_date = pd.Timestamp(year=today.year, month=1, day=1)
        end_date = today if today <= max_date else max_date
    elif preset == "Last 30 Days":
        start_date = max(today - pd.Timedelta(days=30), min_date)
        end_date = today if today <= max_date else max_date
    elif preset == "Last 60 Days":
        start_date = max(today - pd.Timedelta(days=60), min_date)
        end_date = today if today <= max_date else max_date
    elif preset == "Last 90 Days":
        start_date = max(today - pd.Timedelta(days=90), min_date)
        end_date = today if today <= max_date else max_date
    elif preset == "Last Year":
        prev_year = today.year - 1
        start_date = pd.Timestamp(year=prev_year, month=1, day=1)
        end_date = pd.Timestamp(year=prev_year, month=12, day=31)
        # clamp to data bounds
        start_date = max(start_date, min_date)
        end_date   = min(end_date, max_date)
    elif preset == "Today Only":
        # if today's not in data, clamp to nearest
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
            max_value=max_date
        )
        if start_date > end_date:
            st.error("❗ Start date must be before end date.")
            st.stop()

    # Show chosen range
    st.write(f"**Using data from {start_date.date()} to {end_date.date()}**")

    # Filter daily series
    mask = (daily["Date Closed"] >= pd.to_datetime(start_date)) & \
           (daily["Date Closed"] <= pd.to_datetime(end_date))
    daily = daily.loc[mask].reset_index(drop=True)

    # Recompute equity and win/loss flags
    daily["Equity"] = daily["P/L"].cumsum()
    daily["Win"]    = daily["P/L"] > 0
    daily["Loss"]   = daily["P/L"] < 0

    # Compute streaks
    def compute_streaks(series):
        return [len(list(group)) for key, group in groupby(series) if key]

    win_streaks  = compute_streaks(daily["Win"])
    loss_streaks = compute_streaks(daily["Loss"])

    # Metrics
    avg_win_streak    = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    avg_loss_streak   = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
    longest_win_streak  = max(win_streaks)  if win_streaks else 0
    longest_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win            = daily.loc[daily["Win"], "P/L"].mean() if daily["Win"].any() else 0
    avg_loss           = daily.loc[daily["Loss"], "P/L"].mean() if daily["Loss"].any() else 0

    # Drawdowns
    daily["RunningPeak"] = daily["Equity"].cummax()
    daily["Drawdown"]    = daily["RunningPeak"] - daily["Equity"]
    max_drawdown         = daily["Drawdown"].max()
    current_drawdown     = daily["Drawdown"].iloc[-1]
    last_ath_idx         = daily["Equity"].cummax().idxmax()
    days_in_dd           = len(daily) - last_ath_idx - 1

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
else:
    st.info("Please upload a CSV trade log to begin.")
