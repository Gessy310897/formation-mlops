import mlflow.pyfunc
import pandas as pd

class ReadmissionPredictor:
    def __init__(self, model_uri="models:/xgb-readmission/8"):
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, input_df: pd.DataFrame):
        return self.model.predict(input_df)
