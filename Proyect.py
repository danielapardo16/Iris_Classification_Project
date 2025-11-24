import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris 
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Iris Species Classification", layout="wide")
st.title("Iris Species Classification using Logistic Regression and Random Forest")
st.sidebar.title("Integrantes del equipo")
st.sidebar.write("Daniela Pardo")
st.sidebar.write("Juan Donado")

# Cargar el conjunto de datos Iris

@st.cache_data
def  loadData():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df['target_name'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})
    return df, iris 

# Entrenamiento y evaluación de modelos

@st.cache_data
def train_and_evaluate_models(df, iris):
    X = df[iris.feature_names].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1 Score": f1_score(y_test, y_pred, average='macro')
        }

    return results, models
df, iris = loadData()
results, models = train_and_evaluate_models(df, iris)

# Prediccion del usuario

st.header("Make a Prediction")

col1, col2 = st.columns(2)

with col1: 
     sl = st.number_input("Sepal length (cm)", value=5.1)
     sw = st.number_input("Sepal width (cm)", value=3.5)
     pl = st.number_input("Petal length (cm)", value=1.4)
     pw = st.number_input("Petal width (cm)", value=0.2)

with col2:
    model_choice = st.selectbox("Select the model", list(models.keys()))

if st.button("Predict Species"):
    X_new = np.array([[sl, sw, pl, pw]])
    pred_class = models[model_choice].predict(X_new)[0]
    st.success(f"Predicted species: **{iris.target_names[pred_class]}**")

    # Scatter 3D with new point
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'green', 'blue']

    for i, target_name in enumerate(iris.target_names):
        subset = df[df['target'] == i]
        ax.scatter(
            subset[iris.feature_names[0]],
            subset[iris.feature_names[1]],
            subset[iris.feature_names[2]],
            c=colors[i], label=target_name, alpha=0.6
        )

    ax.scatter(sl, sw, pl, c="black", s=120, marker="X", label="Your point")

    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_zlabel(iris.feature_names[2])
    ax.set_title("3D Plot with Your Prediction")
    ax.legend()

    st.pyplot(fig)

# Metrics Comparison

st.header("Model Performance Comparison")

results_df = pd.DataFrame(results).T
st.dataframe(results_df)
st.bar_chart(results_df)

# Visualizaciones 

st.header("3D Visualization of the Iris Dataset")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'green', 'blue']

for i, target_name in enumerate(iris.target_names):
    subset = df[df['target'] == i]
    ax.scatter(
        subset[iris.feature_names[0]],
        subset[iris.feature_names[1]],
        subset[iris.feature_names[2]],
        c=colors[i], label=target_name
    )

ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])
ax.set_title("3D Scatter Plot of Iris Dataset")
ax.legend()

st.pyplot(fig)

# Histograms of features

st.header("Feature Histograms")

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
axs = axs.flatten()

for i, feature in enumerate(iris.feature_names):
    axs[i].hist(df[feature], bins=15, color='lightblue', edgecolor='black')
    axs[i].set_title(feature)

st.pyplot(fig)

# Dataset Preview and Description

st.header("Dataset Preview")
st.dataframe(df.head())

with st.expander("Dataset Description"):
    st.text(iris.DESCR)






