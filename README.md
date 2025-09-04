# Delivery-Time-Prediction
This repository contains a machine learning project for predicting delivery times based on order and delivery-related features. It covers data preprocessing, exploratory analysis, model training, evaluation, and deployment-ready scripts to help optimize logistics and improve customer experience.


# MLOps

Build an ML pipeline to predict delivery times using regression models such as Linear Regression, Random Forest, XGBoost, and Gradient Boosting. Use DVC, Git, MLflow, Docker, GitHub Actions, FastAPI, Streamlit, Kubernetes, Prometheus & Grafana for end-to-end MLOps.

The Delivery-Time-Prediction-MLOps project focuses on building and deploying a machine learning pipeline to estimate delivery times for orders. The system predicts delivery duration based on features like order details, distance, delivery partner attributes, and other contextual factors.

The pipeline begins with data preprocessing, including missing value handling, outlier treatment, feature scaling, and encoding categorical variables. Various regression algorithms such as Linear Regression, Random Forest, and XGBoost will be trained and compared to select the best-performing model.

To integrate MLOps principles, the project uses:

- **DVC + Git** for dataset and code versioning.  
- **MLflow** for experiment tracking and model registry.  
- **Docker** for containerization of the ML pipeline.  
- **GitHub Actions** for CI/CD automation.  
- **FastAPI** to serve the trained model as an API.  
- **Streamlit** to provide an interactive user interface for predictions.  
- **Kubernetes** for orchestration and scalability.  
- **Prometheus & Grafana** for monitoring and visualization of deployed services.
