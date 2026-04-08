import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Weather Analysis", layout="wide")

# ── Header ───────────────────────────────────────────────────
st.title("Weather Data Analysis")
st.markdown("**250 Daily Observations — January to September 2024**")
st.divider()

# ── Load Data ────────────────────────────────────────────────
data = pd.read_excel("weather_250.xlsx", sheet_name="Weather Data")
data.rename(columns={"Date":"date","Temperature (C)":"temp","Humidity (%)":"humidity",
    "Wind Speed (km/h)":"wind_speed","Pressure (hPa)":"pressure","Rainfall (mm)":"rainfall"}, inplace=True)
data["date"] = pd.to_datetime(data["date"])
data["month"] = data["date"].dt.month
data["month_name"] = data["date"].dt.strftime("%b")
data["rain"] = data["humidity"] > 70
data["temp_7day_avg"] = data["temp"].rolling(7).mean()
data["humidity_7day_avg"] = data["humidity"].rolling(7).mean()

z = (data["temp"].to_numpy() - np.mean(data["temp"])) / np.std(data["temp"])
data["zscore"] = z
anomalies = data[np.abs(z) > 2]

columns = ["temp","humidity","wind_speed","pressure","rainfall"]
corr_matrix = np.corrcoef(data[columns].to_numpy().T)

monthly_avg = data.groupby("month")[["temp","humidity","wind_speed","rainfall"]].mean().round(2)
monthly_avg.index = data.groupby("month")["month_name"].first()
monthly_rain = data.groupby("month_name")["rainfall"].sum().reindex(
    [m for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"] if m in data["month_name"].values])

# ── Stats Row ────────────────────────────────────────────────
st.subheader("Key Statistics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Avg Temperature", f"{data['temp'].mean():.1f} C")
col2.metric("Avg Humidity", f"{data['humidity'].mean():.1f} %")
col3.metric("Avg Wind Speed", f"{data['wind_speed'].mean():.1f} km/h")
col4.metric("Rainy Days", f"{data['rain'].sum()} / 250")
col5.metric("Anomalies", f"{len(anomalies)}")
st.divider()

# ── Raw Data Table ───────────────────────────────────────────
with st.expander("View Raw Dataset"):
    st.dataframe(data[["date","temp","humidity","wind_speed","pressure","rainfall","rain"]], use_container_width=True)

st.divider()

# ── Figure 1: Temperature ────────────────────────────────────
st.subheader("Act I — Temperature Trend")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(data["date"], data["temp"], color="#E63946", alpha=0.4, lw=1, label="Daily Temp")
ax1.plot(data["date"], data["temp_7day_avg"], color="#E63946", lw=2, label="7-Day Avg")
ax1.scatter(anomalies["date"], anomalies["temp"], color="black", s=50, zorder=5, label="Anomaly")
ax1.set_title("Temperature Trend (C)")
ax1.legend()
plt.tight_layout()
st.pyplot(fig1)

# ── Figure 2: Humidity ───────────────────────────────────────
st.subheader("Act II — Humidity & Rain Prediction")
col_a, col_b = st.columns(2)

with col_a:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(data["date"], data["humidity"], color="#457B9D", alpha=0.4, lw=1)
    ax2.plot(data["date"], data["humidity_7day_avg"], color="#457B9D", lw=2)
    ax2.axhline(70, color="red", linestyle="--", lw=1.5, label="Threshold (70%)")
    ax2.set_title("Humidity Trend (%)")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

with col_b:
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    rain_counts = data["rain"].value_counts()
    ax3.pie(rain_counts, labels=["Rainy","Dry"] if rain_counts.index[0] else ["Dry","Rainy"],
            colors=["#457B9D","#E76F51"], autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2})
    ax3.set_title("Rain Distribution")
    plt.tight_layout()
    st.pyplot(fig3)

st.divider()

# ── Figure 3: Monthly ────────────────────────────────────────
st.subheader("Act III — Monthly Aggregation")
col_c, col_d = st.columns(2)
x = np.arange(len(monthly_avg))
ml = monthly_avg.index.tolist()

with col_c:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.bar(x, monthly_avg["temp"], color="#E63946", edgecolor="white", width=0.6)
    ax4.set_title("Avg Temperature by Month (C)")
    ax4.set_xticks(x); ax4.set_xticklabels(ml)
    plt.tight_layout()
    st.pyplot(fig4)

with col_d:
    fig5, ax5 = plt.subplots(figsize=(7, 4))
    ax5.bar(x, monthly_rain.values, color="#264653", edgecolor="white", width=0.6)
    ax5.set_title("Total Rainfall by Month (mm)")
    ax5.set_xticks(x); ax5.set_xticklabels(ml)
    plt.tight_layout()
    st.pyplot(fig5)

col_e, col_f = st.columns(2)

with col_e:
    fig6, ax6 = plt.subplots(figsize=(7, 4))
    ax6.bar(x, monthly_avg["humidity"], color="#457B9D", edgecolor="white", width=0.6)
    ax6.axhline(70, color="red", linestyle="--", lw=1.2, label="Threshold")
    ax6.set_title("Avg Humidity by Month (%)")
    ax6.set_xticks(x); ax6.set_xticklabels(ml)
    ax6.legend()
    plt.tight_layout()
    st.pyplot(fig6)

with col_f:
    fig7, ax7 = plt.subplots(figsize=(7, 4))
    ax7.bar(x, monthly_avg["wind_speed"], color="#2A9D8F", edgecolor="white", width=0.6)
    ax7.set_title("Avg Wind Speed by Month (km/h)")
    ax7.set_xticks(x); ax7.set_xticklabels(ml)
    plt.tight_layout()
    st.pyplot(fig7)

st.divider()

# ── Correlation Heatmap ──────────────────────────────────────
st.subheader("Correlation Heatmap")
fig8, ax8 = plt.subplots(figsize=(7, 5))
im = ax8.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax8)
ax8.set_xticks(np.arange(5)); ax8.set_yticks(np.arange(5))
ax8.set_xticklabels(columns, rotation=45, ha="right")
ax8.set_yticklabels(columns)
for i in range(5):
    for j in range(5):
        ax8.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(corr_matrix[i,j]) > 0.5 else "black")
plt.tight_layout()
st.pyplot(fig8)

st.divider()

# ── Anomalies Table ──────────────────────────────────────────
st.subheader("Temperature Anomalies (Z-Score > 2)")
st.dataframe(anomalies[["date","temp","zscore"]].reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.caption("Kerthivaasan Noble Reddy  |  Mohammad Aayan")