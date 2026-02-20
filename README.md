# prosperity-prognosticator-machine-learning-for-startup-success-prediction-
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from preprocessing import build_preprocessor

# Load dataset
df = pd.read_csv("../data/startups.csv")

X = df.drop("success", axis=1)
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline
preprocessor = build_preprocessor(df)

model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Hyperparameter tuning
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)

best_model = grid_search.best_estimator_

# Evaluate
y_probs = best_model.predict_proba(X_test)[:,1]
roc_score = roc_auc_score(y_test, y_probs)
print("Test ROC-AUC:", roc_score)

# Save model
joblib.dump(best_model, "../models/startup_pipeline.pkl")
