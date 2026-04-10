import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fake News Dashboard", layout="wide")

API_URL = "https://fake-news-detector-tke1.onrender.com/predict"
HISTORY_URL = "https://fake-news-detector-tke1.onrender.com/history"

st.title("📰 Fake News Detection Dashboard")
st.write("DistilBERT + FastAPI + MongoDB System")

# ===================== SIDEBAR =====================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "History", "Analytics"])


# ===================== PREDICT =====================
if page == "Predict":
    st.header("🔍 Predict News")

    text = st.text_area("Enter news text")

    if st.button("Predict"):
        if text.strip() == "":
            st.warning("Please enter text")
        else:
            try:
                response = requests.post(API_URL, json={"text": text})

                if response.status_code == 200:
                    result = response.json()

                    st.success(f"Prediction: {result['label']}")
                    st.info(f"Confidence: {result['confidence']:.4f}")

                else:
                    st.error("API error")

            except Exception as e:
                st.error(f"Connection error: {e}")


# ===================== HISTORY =====================
elif page == "History":
    st.header("📊 Prediction History")

    if st.button("Load History"):
        try:
            response = requests.get(HISTORY_URL)

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)

                if len(df) > 0:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No data available")

            else:
                st.error("Failed to fetch history")

        except Exception as e:
            st.error(f"Connection error: {e}")


# ===================== ANALYTICS =====================
elif page == "Analytics":
    st.header("📈 Analytics Dashboard")

    try:
        response = requests.get(HISTORY_URL)

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)

            if len(df) > 0:

                # --------- PIE CHART ----------
                st.subheader("Fake vs Real Distribution")

                counts = df["label"].value_counts()

                fig1, ax1 = plt.subplots()
                ax1.pie(counts, labels=counts.index, autopct="%1.1f%%")
                st.pyplot(fig1)

                # --------- CONFIDENCE ----------
                st.subheader("Confidence Distribution")

                fig2, ax2 = plt.subplots()
                ax2.hist(df["confidence"], bins=10)
                st.pyplot(fig2)

            else:
                st.warning("No data available yet")

        else:
            st.error("Failed to load analytics")

    except Exception as e:
        st.error(f"Connection error: {e}")
