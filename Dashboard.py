import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

def Weather_analysis(df):
  df_analysis = pd.DataFrame(df)
  sns.set(style="whitegrid", palette="Set2")
  fig, ax = plt.subplots(figsize=(12,8))
  sns.violinplot(x="weathersit", y="cnt", data=df_analysis, ax=ax, palette="Set2", inner="quartile")
  mean_points = df_analysis.groupby("weathersit")["cnt"].mean().values
  ax.scatter(x=np.arange(len(mean_points)), y=mean_points, color="red", s=100, marker="o", label="Mean")
  ax.set_title("Analisis sepeda berdasarkan weathers",  fontsize=16)
  ax.set_xlabel("Weathers", fontsize=14)
  ax.set_ylabel("Count sepeda", fontsize=14)
  ax.legend(title="Weather Mean", loc="upper right")

  return fig,ax

def hours_prediction(df):
  X = df[["hr"]]
  y = df["cnt"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  poly = PolynomialFeatures(degree=9)
  X_train_poly = poly.fit_transform(X_train)
  X_test_poly = poly.transform(X_test)
  model = LinearRegression()
  model.fit(X_train_poly, y_train)
  y_pred = model.predict(X_test_poly)
  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean Squared Error: {mse}")
  fig, ax = plt.subplots(figsize=(10,6))
  plt.scatter(X_test, y_test, color="black", label="Actual Data")
  X_test_sorted, y_pred_sorted = zip(*sorted(zip(X_test.values, y_pred)))
  plt.plot(X_test_sorted, y_pred_sorted, color='blue', linewidth = 3, label = f"Polynomial Regression")
  plt.title(f"Prediksi demografi peminjaman sepeda")
  plt.xlabel("Hour")
  plt.ylabel("Count")
  plt.legend()

  return fig, ax

day_df = pd.read_csv("https://raw.githubusercontent.com/Avent001/Dashboard-dicoding/master/Bike-sharing-dataset/day.csv")
hour_df = pd.read_csv("https://raw.githubusercontent.com/Avent001/Dashboard-dicoding/master/Bike-sharing-dataset/hour.csv")
st.sidebar.title("Bike Sharing Analysis Dashboard")

page = st.sidebar.radio("Select a page", ["Weather Analysis", "Hourly Prediction"])

if page == "Weather Analysis":
    st.subheader("Weather Analysis")
    fig, _ = Weather_analysis(day_df)
    st.pyplot(fig)

elif page == "Hourly Prediction":
    st.subheader("Hourly Bike Demand Prediction")
    fig, _ = hours_prediction(hour_df)
    st.pyplot(fig)
