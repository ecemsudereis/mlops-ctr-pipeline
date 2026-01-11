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

SAMPLE_SIZE = 10000  
DATA_PATH = 'train.csv' 
CHECKPOINT_DIR = 'checkpoints'
MODEL_NAME = 'ctr_model_v1.pkl'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class ResilienceCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        if epoch % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch}.json')
            model.save_model(ckpt_path)
            print(f"[Resilience] Checkpoint saved: {ckpt_path}")
        return False

def mock_preprocessing(path):
    print("--- Warning: Using Mock Preprocessing ---")
    
    if not os.path.exists(path):
        print("Data file not found, generating RANDOM data...")
        df = pd.DataFrame(np.random.randint(0, 100, size=(SAMPLE_SIZE, 10)), columns=[f'col_{i}' for i in range(10)])
        df['click'] = np.random.randint(0, 2, size=SAMPLE_SIZE) 
    else:
        df = pd.read_csv(path, nrows=SAMPLE_SIZE)

    if 'click' in df.columns:
        y = df['click']
        X = df.drop(['click', 'id'], axis=1, errors='ignore')
    else:
        y = pd.Series(np.random.randint(0, 2, size=len(df)))
        X = df.copy()

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            
    return X, y

def train():
    mlflow.set_experiment("CTR_Prediction_Project")

    with mlflow.start_run():
        X, y = mock_preprocessing(DATA_PATH)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        if num_pos == 0: num_pos = 1 
        
        scale_pos_weight = num_neg / num_pos
        print(f"Calculated Imbalance Ratio: {scale_pos_weight:.2f}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',         
            'max_depth': 5,
            'learning_rate': 0.1,
            'scale_pos_weight': scale_pos_weight, 
            'seed': 42
        }
        
        mlflow.log_params(params) 

        print("Starting model training...")
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=50, 
            evals=[(dtrain, 'train'), (dtest, 'test')],
            callbacks=[ResilienceCallback()], 
            verbose_eval=10
        )

        preds = model.predict(dtest)
        auc = roc_auc_score(y_test, preds)
        print(f"--- FINAL TEST AUC: {auc:.4f} ---")
        
        mlflow.log_metric("auc", auc)
        
        joblib.dump(model, MODEL_NAME)
        mlflow.xgboost.log_model(model, "model")
        print("Model and logs saved successfully.")

if __name__ == "__main__":
    train()