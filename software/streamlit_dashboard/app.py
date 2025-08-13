# software/streamlit_dashboard/app.py
import streamlit as st
import pandas as pd, os, joblib
import matplotlib.pyplot as plt
from translate import Translator

# Translator for offline translation
translator = Translator(to_lang="en")

# List of official languages as per Eighth Schedule
translations_meta = {
    "en": "English (English)",
    "hi": "हिन्दी (Hindi)",
    "ta": "தமிழ் (Tamil)",
    "as": "অসমীয়া (Assamese)",
    "bn": "বাংলা (Bengali)",
    "brx": "बड़ो (Bodo)",
    "doi": "डोगरी (Dogri)",
    "gu": "ગુજરાતી (Gujarati)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "ks": "كشميري (Kashmiri)",
    "kok": "कोंकणी (Konkani)",
    "mai": "मैथिली (Maithili)",
    "ml": "മലയാളം (Malayalam)",
    "mni": "মেইতেই (Manipuri)",
    "mr": "मराठी (Marathi)",
    "ne": "नेपाली (Nepali)",
    "or": "ଓଡ଼ିଆ (Odia)",
    "pa": "ਪੰਜਾਬੀ (Punjabi)",
    "sa": "संस्कृतम् (Sanskrit)",
    "sat": "ᱥᱟᱱᱛᱟᱲᱤ (Santali)",
    "sd": "سنڌي (Sindhi)",
    "te": "తెలుగు (Telugu)",
    "ur": "اُردُو (Urdu)"
}

# Manual translations for some languages
translations = {
    "en": {
        "title": "SmartFab Demo Dashboard",
        "recent_readings": "Recent Readings & Predictions",
        "quick_stats": "Quick Stats",
        "mean_current": "Mean Current (A)",
        "mean_temp": "Mean Temp (°C)",
        "mean_vib": "Mean Vibration",
        "model_loaded": "Model loaded",
        "anomaly_counts": "Anomaly Counts by Machine",
        "power_vs_solar": "Power vs Solar (last 200)",
        "no_data": "No data available",
        "no_model": "Model not found. Train model first.",
        "note": "Note: Demo uses synthetic data to validate pipeline. Live deployment collects local data on edge devices."
    },
    "hi": {
        "title": "स्मार्टफैब डेमो डैशबोर्ड",
        "recent_readings": "हाल की रीडिंग और पूर्वानुमान",
        "quick_stats": "त्वरित आँकड़े",
        "mean_current": "औसत करंट (A)",
        "mean_temp": "औसत तापमान (°C)",
        "mean_vib": "औसत कंपन",
        "model_loaded": "मॉडल लोड हुआ",
        "anomaly_counts": "मशीन द्वारा विसंगति गणना",
        "power_vs_solar": "पावर बनाम सोलर (अंतिम 200)",
        "no_data": "डेटा उपलब्ध नहीं है",
        "no_model": "मॉडल नहीं मिला। पहले मॉडल ट्रेन करें।",
        "note": "नोट: डेमो पाइपलाइन को सत्यापित करने के लिए सिंथेटिक डेटा का उपयोग करता है। लाइव डिप्लॉयमेंट एज डिवाइस पर स्थानीय डेटा एकत्र करता है।"
    },
    "ta": {
        "title": "ஸ்மார்ட்ஃபாப் டெமோ டாஷ்போர்டு",
        "recent_readings": "சமீபத்திய பதிவுகள் & கணிப்புகள்",
        "quick_stats": "விரைவான புள்ளிவிவரங்கள்",
        "mean_current": "சராசரி மின்சாரம் (A)",
        "mean_temp": "சராசரி வெப்பநிலை (°C)",
        "mean_vib": "சராசரி அதிர்வு",
        "model_loaded": "மாதிரி ஏற்றப்பட்டது",
        "anomaly_counts": "இயந்திர வாரியாக கோளாறுகள்",
        "power_vs_solar": "பவர் Vs சோலார் (கடைசி 200)",
        "no_data": "தரவு இல்லை",
        "no_model": "மாதிரி கிடைக்கவில்லை. முதலில் மாதிரி பயிற்சி செய்யவும்.",
        "note": "குறிப்பு: டெமோ செயற்கை தரவைப் பயன்படுத்துகிறது. நேரடி பயன்பாடு எட்ஜ் சாதனங்களில் உள்ளூர் தரவை சேகரிக்கும்."
    }
}

# Fill missing translations automatically
for lang_code in translations_meta:
    if lang_code not in translations:
        translations[lang_code] = {}
        for key, text in translations["en"].items():
            try:
                translated_text = translator.translate(text, dest=lang_code)
            except:
                translated_text = f"[Translation for {lang_code}]"
            translations[lang_code][key] = translated_text

# Language selector
lang = st.sidebar.selectbox("🌐 Language", list(translations_meta.keys()), format_func=lambda x: translations_meta[x])
T = translations[lang]

# Data paths
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE, 'data', 'synthetic_sensor_data.csv')
MODEL_PATH = os.path.join(BASE, 'firmware', 'edge_ai_code', 'model.joblib')

st.set_page_config(page_title=T["title"], layout='wide')
st.title(T["title"])

# Layout
col1, col2 = st.columns([2,1])
with col1:
    st.header(T["recent_readings"])
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').tail(200)
        st.dataframe(df.tail(20))
    else:
        st.info(T["no_data"])

with col2:
    st.header(T["quick_stats"])
    if os.path.exists(DATA_PATH):
        df_all = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
        st.metric(T["mean_current"], round(df_all['current_a'].mean(),3))
        st.metric(T["mean_temp"], round(df_all['temperature_c'].mean(),3))
        st.metric(T["mean_vib"], int(df_all['vibration_raw'].mean()))
    else:
        st.write(T["no_data"])

# Model predictions
if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
    model = joblib.load(MODEL_PATH)
    st.success(T["model_loaded"])
    features = ['current_a','temperature_c','vibration_raw','power_kw','solar_kw','current_ma_3','temp_ma_3','vib_ma_3']
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp']).sort_values('timestamp')
    df['current_ma_3'] = df.groupby('machine_id')['current_a'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['temp_ma_3'] = df.groupby('machine_id')['temperature_c'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['vib_ma_3'] = df.groupby('machine_id')['vibration_raw'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    X = df[features].astype(float)
    preds = model.predict(X)
    scores = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else [None]*len(preds)
    df['anomaly_pred'] = preds
    df['anomaly_score'] = scores
    st.subheader(T["anomaly_counts"])
    st.bar_chart(df.groupby('machine_id')['anomaly_pred'].sum())
    st.subheader(T["power_vs_solar"])
    plt.figure(figsize=(8,3))
    plt.plot(df['timestamp'].tail(200), df['power_kw'].tail(200), label='power_kw')
    plt.plot(df['timestamp'].tail(200), df['solar_kw'].tail(200), label='solar_kw')
    plt.legend()
    st.pyplot(plt)
else:
    st.warning(T["no_model"])

st.markdown('---')
st.markdown(f"**{T['note']}**")
