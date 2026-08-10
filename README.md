# GridGuard AI

## Evaluating Regional Power Grid Supply Risk Using Geographic K-Means Clustering with Random Forest Risk Classification

GridGuard AI is a machine learning project developed to evaluate regional power-grid supply risk. The system combines K-Means clustering and Random Forest classification to identify and predict Low, Medium, and High risk conditions.

## 🌐 Live Website

[GridGuard AI](https://gridguardai.streamlit.app/)

---

## ❗ Problem

Power-grid conditions can vary across different regions due to changes in power demand, energy availability, shortages, and grid deviations. Identifying regions with higher supply risk from these multiple factors can be difficult using traditional analysis alone.

The dataset also does not directly provide a predefined Low, Medium, or High risk label for every observation. Therefore, a method is required to first identify similar grid conditions and then classify those conditions into meaningful risk categories.

---

## 📌 Problem Statement

**"Evaluating regional power grid supply risk by combining geographic K-Means clustering with Random Forest risk classification."**

The problem is to develop a machine learning system that can:

- Identify similar regional power-grid conditions using K-Means clustering.
- Analyze the characteristics of each cluster using centroid values.
- Assign meaningful Low, Medium, and High risk levels to the clusters.
- Train a Random Forest model to classify power-grid conditions into the corresponding risk levels.
- Provide risk predictions for new grid observations through a Streamlit web application.

---

## 🎯 Objective

The main objective of GridGuard AI is to develop an intelligent system for evaluating regional power-grid supply risk using geographical and operational grid features.

The project aims to combine:

- **K-Means** for discovering similar grid conditions.
- **Centroid analysis** for understanding cluster characteristics.
- **Risk scoring** for assigning risk levels.
- **Random Forest** for predicting risk categories.
- **Streamlit** for deploying the system as a web application.

---

## 🔄 Project Workflow

Power Grid Dataset  
↓  
Data Preprocessing  
↓  
Feature Selection  
↓  
Feature Scaling  
↓  
K-Means Clustering  
↓  
Cluster Centroid Analysis  
↓  
Risk Score Calculation  
↓  
Low / Medium / High Risk  
↓  
Random Forest Classification  
↓  
Model Evaluation  
↓  
Streamlit Deployment

---

## 📊 Features Used

The following nine features are used for clustering and risk classification:

- Latitude
- Longitude
- Max Demand Met
- Shortage During Peak
- Energy Met
- Drawl Schedule
- OD(+) / UD(-)
- Max OD
- Energy Shortage

---

## 🤖 Machine Learning Models

### 1. K-Means Clustering

K-Means is an unsupervised machine learning algorithm used to group similar power-grid observations into clusters.

In this project, three clusters are created. The centroid characteristics of these clusters are then analyzed to determine their risk levels.

### 2. Random Forest Classification

Random Forest is a supervised machine learning algorithm used to classify the grid conditions into:

- Low Risk
- Medium Risk
- High Risk

The nine grid features are used as input variables, and the generated risk category is used as the target.

---

## 📈 Model Evaluation

The Random Forest model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training vs Testing Accuracy
- Five-Fold Cross-Validation

The model achieved approximately **99.67% testing accuracy**.

The five-fold cross-validation produced a mean F1-score of approximately **0.994**.

---

## 🖥️ Streamlit Application

The trained machine learning models are deployed using Streamlit.

The application allows users to enter power-grid parameters and obtain a predicted risk category.

### Prediction Flow

User Input  
↓  
Feature Scaling  
↓  
K-Means Clustering  
↓  
Cluster-to-Risk Mapping  
↓  
Random Forest Prediction  
↓  
Risk Result

---

## 📁 Project Structure

```text
gridguard-ai/
│
├── app.py
├── GridGuard_AI.ipynb
├── power_grid.csv
├── gridguard_random_forest.pkl
├── gridguard_kmeans.pkl
├── gridguard_scaler.pkl
├── cluster_to_risk.pkl
├── requirements.txt
└── README.md
