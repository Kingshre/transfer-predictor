import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="CA Transfer Outcome Predictor", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(APP_DIR, 'transfer_clean.csv'))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(APP_DIR, 'model.pkl'))

df = load_data()
model = load_model()

# Encoders
le_college = LabelEncoder().fit(df['localeName'])
le_ethnicity = LabelEncoder().fit(df['subgroup'])

st.title("🎓 CA Transfer Outcome Predictor")
st.markdown("Predicting UC/CSU transfer equity gaps across California community colleges using CCCCO data.")

# Sidebar
st.sidebar.header("Filters")
year = st.sidebar.selectbox("Academic Year", sorted(df['academicYear'].unique(), reverse=True))
college = st.sidebar.multiselect("College", sorted(df['localeName'].unique()), default=[])

filtered = df[df['academicYear'] == year]
if college:
    filtered = filtered[filtered['localeName'].isin(college)]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered))
col2.metric("At-Risk Groups", filtered['at_risk'].sum())
col3.metric("Avg Transfer Rate", f"{filtered['transfer_rate'].mean()*100:.1f}%")

# Bar chart by ethnicity
st.subheader("📊 Transfer Rate by Ethnicity")
eth_avg = filtered.groupby('subgroup')['transfer_rate'].mean().reset_index()
eth_avg.columns = ['Ethnicity', 'Transfer Rate']
eth_avg = eth_avg.sort_values('Transfer Rate', ascending=True)
fig = px.bar(eth_avg, x='Transfer Rate', y='Ethnicity', orientation='h',
             color='Transfer Rate', color_continuous_scale='RdYlGn',
             height=400)
st.plotly_chart(fig, use_container_width=True)

# Transfer rate by college
st.subheader("🏫 Transfer Rate by College")
col_avg = filtered.groupby('localeName')['transfer_rate'].mean().reset_index()
col_avg.columns = ['College', 'Transfer Rate']
col_avg = col_avg.sort_values('Transfer Rate', ascending=True)
fig2 = px.bar(col_avg, x='Transfer Rate', y='College', orientation='h',
              color='Transfer Rate', color_continuous_scale='RdYlGn',
              height=500)
st.plotly_chart(fig2, use_container_width=True)

# SHAP
st.subheader("🔍 Feature Importance (SHAP)")
shap_img = Image.open(os.path.join(APP_DIR, 'shap_summary.png'))
st.image(shap_img, use_container_width=True)

# Risk table
st.subheader("📋 At-Risk Groups Table")
st.dataframe(
    filtered[['localeName', 'subgroup', 'academicYear', 'transfer_rate', 'at_risk']]
    .sort_values('transfer_rate')
    .reset_index(drop=True),
    use_container_width=True
)

# Prediction tool
st.subheader("⚡ Predict Transfer Risk")
col1, col2 = st.columns(2)
with col1:
    college_input = st.selectbox("College", sorted(df['localeName'].unique()))
    ethnicity_input = st.selectbox("Ethnicity", sorted(df['subgroup'].unique()))
with col2:
    year_input = st.selectbox("Year", sorted(df['academicYear'].unique()))
    denom_input = st.number_input("Student Group Size", min_value=1, value=200)

if st.button("Predict Risk"):
    input_data = np.array([[
        le_college.transform([college_input])[0],
        le_ethnicity.transform([ethnicity_input])[0],
        year_input,
        denom_input
    ]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ AT RISK — {prob*100:.1f}% probability of below-average transfer rate")
    else:
        st.success(f"✅ ON TRACK — {prob*100:.1f}% risk score")