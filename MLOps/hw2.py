import mlflow
import os
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import pandas as pd

FEATURES = ['account_length', 'phone_number',
       'international_plan', 'voice_mail_plan', 'number_vmail_messages',
       'total_day_minutes', 'total_day_calls', 'total_day_charge',
       'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
       'total_night_minutes', 'total_night_calls', 'total_night_charge',
       'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
       'number_customer_service_calls']

model_names = ["log_regression", "random_forest", "desicion_tree"]
models = dict(
    zip(model_names, [
        LogisticRegression(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
    ]))

def main():
    dataset = fetch_openml(name='churn', version=1)
    X, y = dataset["data"][FEATURES], dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    scaler = StandardScaler()
    X_train_prepared = scaler.fit_transform(X_train)
    X_test_prepared = scaler.transform(X_test)
    exp_name = "admoskalenko"
    exps = mlflow.search_experiments(filter_string=f"name = '{exp_name}'")
    if len(exps) > 0:
        experiment_id = exps[0].experiment_id
    else:
        experiment_id = mlflow.create_experiment(exp_name)
    
    mlflow.set_experiment(exp_name)
    
    with mlflow.start_run(run_name="tau_ceti_pn", experiment_id = experiment_id, description = "parent") as parent_run:
        
        X_train_prepared = pd.DataFrame(data=X_train_prepared, columns=FEATURES)
        X_test_prepared = pd.DataFrame(data=X_test_prepared, columns=FEATURES)
        
        for model_name in models.keys():
        # Запустим child run на каждую модель.
            with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, nested=True) as child_run:
                model = models[model_name]
                model.fit(X_train_prepared, y_train)
                prediction = model.predict(X_test_prepared)
                eval_df = X_test_prepared.copy()
                eval_df['labels'] = y_test.to_numpy()
                signature = infer_signature(X_test_prepared, prediction)
                model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)
                mlflow.evaluate(
                    model=model_info.model_uri,
                    data=eval_df,
                    targets="labels",
                    model_type="classifier",
                    evaluators=["default"],
                )
 


if __name__ == '__main__':
    main()