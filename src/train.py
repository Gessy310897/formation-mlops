import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# 📂 Chargement des données
df = pd.read_csv("data/DSA-2025_clean_data.tsv", sep="\t")

# 🎯 Séparation features / cible
X = df.drop(columns=["readmission"])
y = df["readmission"]

# ✂️ Split train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Nouveau Grid Search
param_grid = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1]
}

for max_depth in param_grid["max_depth"]:
    for learning_rate in param_grid["learning_rate"]:
        with mlflow.start_run():
            print(f"Training with max_depth={max_depth}, learning_rate={learning_rate}")
            model = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_metric("accuracy", acc)

            input_example = X_test.iloc[:1]
            signature = infer_signature(X_train, model.predict(X_train))

            mlflow.xgboost.log_model(
                model,
                "model",
                input_example=input_example,
                signature=signature,
                registered_model_name="xgb-readmission"
            )

            print(f"✅ Accuracy: {acc:.4f}")
