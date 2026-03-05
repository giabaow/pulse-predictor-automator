import matplotlib

matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    BaggingClassifier,
    StackingClassifier,
)
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv("Data/HeartDisease.csv")
numerical_cols = ["Age", "MaxHeartRate"]
categorical_cols = ["Sex", "ChestPainType", "ExerciseInducedAngina"]

X = df.drop(["HeartDisease"], axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

num_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


transform = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(), categorical_cols),
        ("num", num_transformer, numerical_cols),
    ]
)

svm_pipeline = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", SVC(probability=True, class_weight="balanced", random_state=42)),
    ]
)

param_grid_svm = {
    "model__kernel": ["linear", "rbf", "poly"],
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", "auto"],
    "model__degree": [2, 3],  # only used for 'poly'
}

grid_svm = GridSearchCV(
    estimator=svm_pipeline, param_grid=param_grid_svm, cv=cv, scoring="f1", n_jobs=-1
)

# Fit GridSearch
grid_svm.fit(X_train, y_train)

cal_svm = CalibratedClassifierCV(estimator=grid_svm.best_estimator_, cv=cv)
cal_svm.fit(X_train, y_train)

# Predict
y_pred_svm = cal_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_svm)
f1 = f1_score(y_test, y_pred_svm, average="macro")
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {accuracy}, F1 Score = {f1}.")


classes = cal_svm.classes_

cm = confusion_matrix(y_test, y_pred_svm, labels=classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)

plt.savefig("Results/model_results.png", dpi=120)
plt.close()

import skops.io as sio

# Save your pipeline
sio.dump(svm_pipeline, "Model/pulse_pipeline.skops")

# Load safely (trusted must be a list of types used in your pipeline)
trusted_types = [
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.svm._classes.SVC",
]

sio.load("Model/pulse_pipeline.skops", trusted=trusted_types)
