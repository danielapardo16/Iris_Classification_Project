# Iris Species Classification — Proyecto Final (Data Mining)

Este repositorio contiene el proyecto final del curso de Data Mining, donde se desarrolla un modelo de clasificación para predecir la especie de flores Iris utilizando técnicas de machine learning y un dashboard interactivo creado con Streamlit.

## Objetivo del proyecto

Entrenar y evaluar dos modelos de clasificación (Logistic Regression y Random Forest) y permitir que el usuario interactúe con el sistema mediante un dashboard para:

- Visualizar métricas de desempeño

- Realizar predicciones ingresando valores de las características del Iris

- Observar gráficos 3D

- Explorar histogramas y el dataset

## Contenido del repositorio

- Proyect.py — Dashboard interactivo en Streamlit

- requirements.txt — Librerías necesarias

- README.md — Documento descriptivo del proyecto

## Metodología (Workflow)

1. **Data Understanding:**
Se cargó el dataset Iris, se analizaron las variables, se identificó el número de muestras, características y clases, y se verificó la ausencia de valores faltantes.

2. **Exploratory Data Analysis (EDA):**
Se generaron histogramas y gráficos 3D para comprender la distribución de las características y las diferencias entre especies.

3. **Preprocesamiento:**
Se realizó la separación del dataset en conjuntos de entrenamiento y prueba (80/20) utilizando estratificación para mantener la proporción de clases.

4. **Modelado:**
Se entrenaron dos modelos de clasificación: Logistic Regression y Random Forest. Ambos fueron ajustados con sus parámetros por defecto para mantener simplicidad y claridad metodológica.

5. **Evaluación:**
Se calcularon las métricas Accuracy, Precision, Recall y F1 Score utilizando el promedio macro para comparar el rendimiento de los modelos.

6. **Dashboard:**
Se construyó un dashboard interactivo en Streamlit que permite:

- Ingresar valores para predecir la especie de la flor

- Visualizar la predicción sobre un gráfico 3D

- Consultar métricas de los modelos

- Explorar histogramas y el dataset

- Revisar la descripción del dataset

## Instrucciones de ejecución

1. Instalar las dependencias necesarias:

pip install -r requirements.txt


2. Ejecutar el dashboard:

streamlit run Proyect.py


En caso de error:

python -m streamlit run Proyect.py

## Integrantes del equipo

- Daniela Pardo

- Juan Donado
