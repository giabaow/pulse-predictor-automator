import gradio as gr
import skops.io as sio
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings from skops
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Trusted types in your pipeline
trusted_types = [
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing.OneHotEncoder",
    "sklearn.preprocessing.StandardScaler",
    "sklearn.compose.ColumnTransformer",
    "sklearn.preprocessing.OrdinalEncoder",
    "sklearn.impute.SimpleImputer",
    "sklearn.ensemble.RandomForestClassifier",
    "numpy.dtype",
]

# Load your trained pipeline
pipe = sio.load("./Model/pulse_pipeline.skops", trusted=trusted_types)

# Prediction function
def predict_heart_disease(age, sex, cp, trestbps, chol):
    """Predict heart disease based on patient features."""
    features = [age, sex, cp, trestbps, chol]
    pred = pipe.predict([features])[0]
    label = "Heart Disease: YES" if pred == 1 else "Heart Disease: NO"
    return label

# Gradio inputs
inputs = [
    gr.Slider(29, 77, step=1, label="Age"),                    # Example min/max from dataset
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio([0, 1, 2, 3], label="Chest Pain Type (cp)"),      # cp encoded
    gr.Slider(94, 200, step=1, label="Resting Blood Pressure"),
    gr.Slider(126, 564, step=1, label="Cholesterol"),
]

# Output
outputs = [gr.Label(num_top_classes=2)]

# Example inputs
examples = [
    [45, "M", 1, 130, 233],
    [60, "F", 3, 145, 300],
    [50, "M", 2, 120, 250],
]

# App text
title = "Heart Disease Prediction"
description = "Enter patient details to predict the likelihood of heart disease."
article = "This app is part of a **CI/CD ML pipeline** demo. It predicts heart disease based on patient clinical data."

# Launch Gradio interface
app = gr.Interface(
    fn=predict_heart_disease,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()

if __name__ == "__main__":
    app.launch()