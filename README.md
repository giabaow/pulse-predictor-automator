# Pulse Predictor Automator
<!-- CI/CD Status Badge -->
[![CI Status](https://img.shields.io/github/actions/workflow/status/giabaow/pulse-predictor-automator/ci.yml?branch=main&style=flat-square&label=CI&logo=github)](https://github.com/giabaow/pulse-predictor-automator/actions)

<!-- License Badge -->
[![License](https://img.shields.io/github/license/giabaow/pulse-predictor-automator?style=flat-square&color=blue)](LICENSE)

## Project Overview

**Pulse Predictor Automator** is a machine learning project designed to predict the likelihood of heart disease based on patient clinical data.  
This project demonstrates **MLOps best practices** by integrating CI/CD workflows, automated training pipelines, model evaluation, and deployment to a Gradio web application.  

It reflects the skills and workflow principles sought for MLOps roles at companies like **AUMOVIO**, including:

- Automating AI model training and evaluation pipelines
- Containerized workflows for reproducible AI experiments
- Efficient management of computational resources
- Integration of Python ML pipelines with modern deployment tools

---

## Features

- End-to-end **ML pipeline** from data preprocessing to prediction
- Supports **model training, evaluation, and deployment** via Makefile automation
- Implements **CI/CD workflows** with GitHub Actions
- Interactive **Gradio web app** for real-time heart disease prediction
- Demonstrates **GPU-aware workflows** and reproducible training pipelines

---
## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pulse-predictor-automator.git
cd pulse-predictor-automator
```
### 2. Create and activate a Python environment
Using conda:
```bash
conda create -n heartdisease-cicd python=3.11 -y
conda activate heartdisease-cicd
```

Or using venv:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```
make install
```
## Usage
### 1. Train the ML pipeline
```
make train
```
Trains a Random Forest + preprocessing pipeline on heart disease data
Outputs metrics to **Results/metrics.txt**

### 2. Evaluate model
```
make eval
```
Generates classification metrics, confusion matrix plots, and report

### 3. Run the Gradio web app locally
```
python App/app.py
```
- Interactive interface for predicting heart disease
- Input features: Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol

## CI/CD Workflow
- Continuous Integration: automated tests, formatting checks, and model training
- Continuous Deployment: pushes trained models and app to Hugging Face Spaces
- Automation is handled via Makefile targets and GitHub Actions workflows
- Demonstrates containerization and reproducibility principles relevant to GPU clusters

## Live Demo
Try the interactive Gradio app locally or deploy it to a public Hugging Face Space for real-time predictions.
[Open Gradio App](https://huggingface.co/spaces/yourusername/pulse-predictor-automator)

## Technical Highlights
- Python-based ML pipeline using scikit-learn, XGBoost, and LightGBM
- Data preprocessing: StandardScaler, OneHotEncoder, SimpleImputer
- Pipeline persistence with skops for reproducibility
- Automated CI/CD workflows for training, evaluation, and deployment
- Headless plotting for GPU/CI-friendly evaluation
- Designed to be compatible with Linux environments, containerized workflows, and CI/CD GPU pipelines

## Skills Demonstrated
- This project demonstrates capabilities that align with the AUMOVIO MLOps role:
- Designing and implementing automated ML pipelines
- Developing CI/CD workflows for AI development
- Debugging and profiling ML model performance
- Deploying ML models and interactive applications
- Following best practices in Python development and software architecture
- Working with virtual environments, containerization, and reproducible ML experiments

## License
This project is licensed under the MIT License.

