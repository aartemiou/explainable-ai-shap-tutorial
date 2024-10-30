# Explainable AI: Trust and Transparency with SHAP

This repository accompanies the *GnoelixiAI Hub Newsletter* edition titled *"Explainable AI: Trust and Transparency with SHAP."* It provides a practical example using SHAP (SHapley Additive exPlanations) to explain predictions from a machine learning model trained on synthetic housing data.

Read the full article on LinkedIn: [Explainable AI: Trust and Transparency with SHAP](https://www.linkedin.com/pulse/explainable-ai-trust-transparency-shap-artemakis-artemiou-iutsf/)

## Contents
- **Data generation and preprocessing**: Creating synthetic housing data with features like square footage, number of bedrooms, and age of home.
- **Random Forest model training**: Training a machine learning model to predict house prices.
- **SHAP values calculation and visualization**: Using SHAP to explain the model's predictions and understand feature importance.

## Prerequisites
Install required libraries with:
```bash
pip install pandas numpy scikit-learn shap matplotlib
