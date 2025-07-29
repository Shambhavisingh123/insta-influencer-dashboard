# 📊 Instagram Influencers Dashboard

This project is a data analysis and machine learning dashboard built using **Streamlit**. It explores a dataset of top Instagram influencers and predicts the **engagement rate** based on followers, influence score, and country.

## 🚀 Features
- Interactive filters (country, followers range)
- Visualizations: bar chart, scatter plot
- Real-time predictions using a trained Random Forest model
- Fully deployable via Streamlit Cloud

## 📁 Files Included
- `app.py`: Main dashboard file
- `top_instagram_influencers.csv`: Dataset file
- `engagement_model.pkl`: Trained ML model
- `engagement_prediction_model.ipynb`: Jupyter notebook for training the model
- `requirements.txt`: Dependencies for running the app
- `README.md`: Project overview

## 🧠 Model Details
- Trained with RandomForestRegressor
- Features used: `followers`, `influence_score`, `country`
- Target: `60_day_eng_rate`

## ▶️ How to Run

1. Clone the repo
2. Create a virtual environment and activate it
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🌐 Deploy
- Push this repo to GitHub
- Connect to [Streamlit Cloud](https://streamlit.io/cloud)
- Deploy your app in minutes

---

📬 Built with ❤️ using Python, Streamlit, and scikit-learn.