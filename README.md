# Pulse Predictor Automator

[![GitHub Actions](https://img.shields.io/github/workflow/status/giabaow/pulse-predictor-automator/Continuous%20Integration)](https://github.com/giabaow/pulse-predictor-automator/actions)  
[![License](https://img.shields.io/github/license/giabaow/pulse-predictor-automator)](LICENSE)

## Project Overview

**Pulse Predictor Automator** is a machine learning project that predicts the likelihood of heart disease based on patient clinical data. The project includes a **CI/CD pipeline** that automates:

- Model training
- Evaluation
- Deployment to Hugging Face Spaces
- Interactive Gradio web app

The Gradio app allows users to input patient information and receive a prediction for heart disease in real time.

---

## Features

- Predicts heart disease based on **age, sex, chest pain type, blood pressure, and cholesterol**.  
- **CI/CD automation**: training, evaluation, and deployment are fully automated using GitHub Actions.  
- Interactive **Gradio web interface** for quick predictions.  
- Headless plotting ensures smooth operation in CI/CD environments.  

---

## Project Structure

