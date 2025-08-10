
# firmware/edge_ai_code/train_model.py
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'synthetic_sensor_data.csv'))
OUT_DIR = os.path.dirname(__file__)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.sort_values(['machine_id','timestamp'], inplace=True)
    df['current_ma_3'] = df.groupby('machine_id')['current_a'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['temp_ma_3'] = df.groupby('machine_id')['temperature_c'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['vib_ma_3'] = df.groupby('machine_id')['vibration_raw'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    features = ['current_a','temperature_c','vibration_raw','power_kw','solar_kw','current_ma_3','temp_ma_3','vib_ma_3']
    df = df.dropna(subset=features+['label'])
    X = df[features].astype(float)
    y = df['label'].astype(int)
    return X,y,df

def train_and_save(random_state=42):
    X,y,df = load_data()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=random_state)
    pipeline = Pipeline([('scaler',StandardScaler()),('clf',RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=random_state,n_jobs=-1))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline,'predict_proba') else None
    report = classification_report(y_test, preds, digits=4)
    auc = roc_auc_score(y_test, proba) if proba is not None else None
    model_path = os.path.join(OUT_DIR, 'model.joblib')
    joblib.dump(pipeline, model_path)
    with open(os.path.join(OUT_DIR,'metrics_summary.txt'),'w') as f:
        f.write('CV F1 mean: {:.4f} std: {:.4f}\\n'.format(cv_scores.mean(), cv_scores.std()))
        if auc is not None:
            f.write('Test ROC AUC: {:.4f}\\n'.format(auc))
        f.write('\\nClassification Report:\\n')
        f.write(report)
    print('Saved model to', model_path)
    print('CV F1 mean:', cv_scores.mean())
    return pipeline

if __name__ == '__main__':
    train_and_save()
