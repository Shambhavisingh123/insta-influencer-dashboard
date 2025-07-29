import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Insta Influencer Insights", page_icon="📱", layout="wide")
st.title("📱 Top Instagram Influencers Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("top_instagram_influencers.csv")
    replace = {'b': 'e9', 'm': 'e6', 'k': 'e3', '%': ''}
    cols = ['followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like', 'total_likes', 'posts']
    df[cols] = df[cols].replace(replace, regex=True).astype(float)
    df.dropna(inplace=True)
    df['country_encoded'] = LabelEncoder().fit_transform(df['country'].astype(str))
    return df

@st.cache_resource
def load_model():
    return joblib.load("engagement_model.pkl")

df = load_data()
model = load_model()

# Sidebar - Filters
st.sidebar.header("📊 Filters")
min_f, max_f = st.sidebar.slider("Followers Range", 
                                  min_value=int(df['followers'].min()), 
                                  max_value=int(df['followers'].max()), 
                                  value=(int(1e6), int(1e8)))
selected_countries = st.sidebar.multiselect("Select Country", df['country'].dropna().unique(),
                                            default=df['country'].dropna().unique())

filtered_df = df[(df['followers'] >= min_f) & (df['followers'] <= max_f)]
if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

# Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Countries by Influencer Count")
    country_counts = filtered_df['country'].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=country_counts.values, y=country_counts.index, ax=ax1, color='orange')
    st.pyplot(fig1)

with col2:
    st.subheader("Followers vs 60-Day Engagement Rate")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=filtered_df, x='followers', y='60_day_eng_rate', hue='country', alpha=0.7, ax=ax2)
    ax2.set_xscale("log")
    st.pyplot(fig2)

# Sidebar - Prediction
st.sidebar.subheader("🔮 Predict Engagement Rate")
followers_input = st.sidebar.number_input("Followers", value=1000000, step=100000)
influence_input = st.sidebar.slider("Influence Score", 0, 100, 75)
country_input = st.sidebar.selectbox("Country", df['country'].unique())

le_country = LabelEncoder()
le_country.fit(df['country'])
country_encoded = le_country.transform([country_input])[0]

# Prediction using a DataFrame (to avoid warning)
input_df = pd.DataFrame([{
    "followers": followers_input,
    "influence_score": influence_input,
    "country_encoded": country_encoded
}])

prediction = model.predict(input_df)
st.sidebar.markdown(f"### 📈 Predicted Engagement Rate: `{round(prediction[0], 2)}%`")

# Data Table
st.subheader("📄 Influencer Data Table")
st.dataframe(filtered_df.head(20))