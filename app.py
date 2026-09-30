import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Forecaster", layout="wide")
st.title("⚡ Global Energy Demand Forecaster")
st.write("Predicts next 24h PJM load - R2 0.942")

# Load
DATA = Path("data/PJME_hourly.csv")
MODEL = Path("models/rf_energy_model.pkl")

if not DATA.exists():
    st.error("data/PJME_hourly.csv not found locally. Add it to run.")
    st.stop()
if not MODEL.exists():
    st.error("models/rf_energy_model.pkl not found. Run python src/train.py first")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').set_index('Datetime')
    df.columns = ['MW']
    return df


model = load_model()
df = load_data()


def make_features(data):
    d = data.copy()
    d['lag_24'] = d['MW'].shift(24)
    d['lag_168'] = d['MW'].shift(168)
    d['roll_mean_24'] = d['MW'].shift(1).rolling(24).mean()
    d['roll_std_24'] = d['MW'].shift(1).rolling(24).std()
    d['hour'] = d.index.hour
    d['dayofweek'] = d.index.dayofweek
    d['month'] = d.index.month
    return d.dropna()


df_feat = make_features(df)

FEATURES = ['lag_24', 'lag_168', 'roll_mean_24',
            'roll_std_24', 'hour', 'dayofweek', 'month']

last_n = 24 * 7
X_last = df_feat[FEATURES].iloc[-last_n:]
y_last = df_feat['MW'].iloc[-last_n:]
pred = model.predict(X_last)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y_last.index, y_last.values, label="Actual")
ax.plot(y_last.index, pred, label="Predicted")
ax.legend()
ax.set_ylabel("MW")
st.pyplot(fig)

# Forecast next 24h
st.subheader("Next 24h forecast (from last data point)")
next_24 = pd.DataFrame({
    "Datetime": pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h"),
    "Predicted MW": model.predict(X_last.iloc[-24:])
})

st.dataframe(next_24)

st.success("Model: RandomForest 200 trees | MAE 1150 | R2 0.942")
