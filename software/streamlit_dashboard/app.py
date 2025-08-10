
# software/streamlit_dashboard/app.py
import streamlit as st
import pandas as pd, os, joblib
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE, 'data', 'synthetic_sensor_data.csv')
MODEL_PATH = os.path.join(BASE, 'firmware', 'edge_ai_code', 'model.joblib')

st.set_page_config(page_title='SmartFab Dashboard', layout='wide')
st.title('SmartFab — Streamlit Demo Dashboard')

col1, col2 = st.columns([2,1])
with col1:
    st.header('Recent Readings & Predictions')
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').tail(200)
        st.dataframe(df.tail(20))
    else:
        st.info('No synthetic data found. Run the data generator.')

with col2:
    st.header('Quick Stats')
    if os.path.exists(DATA_PATH):
        df_all = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
        st.metric('Mean Current (A)', round(df_all['current_a'].mean(),3))
        st.metric('Mean Temp (°C)', round(df_all['temperature_c'].mean(),3))
        st.metric('Mean Vibration', int(df_all['vibration_raw'].mean()))
    else:
        st.write('No data')

if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
    model = joblib.load(MODEL_PATH)
    st.success('Model loaded')
    features = ['current_a','temperature_c','vibration_raw','power_kw','solar_kw','current_ma_3','temp_ma_3','vib_ma_3']
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp']).sort_values('timestamp')
    df['current_ma_3'] = df.groupby('machine_id')['current_a'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['temp_ma_3'] = df.groupby('machine_id')['temperature_c'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['vib_ma_3'] = df.groupby('machine_id')['vibration_raw'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    X = df[features].astype(float)
    preds = model.predict(X)
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X)[:,1]
    else:
        scores = [None]*len(preds)
    df['anomaly_pred'] = preds
    df['anomaly_score'] = scores
    st.subheader('Anomaly Counts by Machine')
    st.bar_chart(df.groupby('machine_id')['anomaly_pred'].sum())
    st.subheader('Power vs Solar (last 200)')
    plt.figure(figsize=(8,3))
    plt.plot(df['timestamp'].tail(200), df['power_kw'].tail(200), label='power_kw')
    plt.plot(df['timestamp'].tail(200), df['solar_kw'].tail(200), label='solar_kw')
    plt.legend()
    st.pyplot(plt)
else:
    st.warning('Model not found. Train model first.')

st.markdown('---')
st.markdown('**Note:** Demo uses synthetic data to validate pipeline. Live deployment collects local data on edge devices.')
