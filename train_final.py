import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.xgboost
import os
import joblib

REAL_DATA_PATH = 'processed_data.parquet'  
CHECKPOINT_DIR = 'checkpoints'
MODEL_NAME = 'ctr_model_final.pkl'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class ResilienceCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        if epoch % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch}.json')
            model.save_model(ckpt_path)
        return False

def load_real_data(path):
    print(f"Loading data from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR: File '{path}' not found!")

    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    
    print(f"Data loaded. Shape: {df.shape}")

    if 'click' not in df.columns:
        raise ValueError("CRITICAL ERROR: 'click' column is missing!")

    y = df['click']
    X = df.drop(['click'], axis=1)

    print("Checking and correcting data types...")
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            print(f" -> Encoding column '{col}' to numeric...")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            
    return X, y

def train():
    mlflow.set_experiment("CTR_Prediction_Final_Production")

    with mlflow.start_run():
        print("--- Final Model Training Started ---")
        
        try:
            X, y = load_real_data(REAL_DATA_PATH)
        except Exception as e:
            print(f"ERROR: {e}")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight = num_neg / num_pos
        print(f"Imbalance Ratio: {scale_pos_weight:.2f}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,          
            'learning_rate': 0.05,
            'scale_pos_weight': scale_pos_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        mlflow.log_params(params)

        print("Starting training with Resilience Mechanism...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            callbacks=[ResilienceCallback()],
            verbose_eval=10
        )

        preds = model.predict(dtest)
        auc = roc_auc_score(y_test, preds)
        print(f"\n✅ --- FINAL AUC SCORE: {auc:.4f} ---")
        
        mlflow.log_metric("auc", auc)
        joblib.dump(model, MODEL_NAME)
        mlflow.xgboost.log_model(model, "model")
        print(f"Model saved: {MODEL_NAME}")

if __name__ == "__main__":
    train()