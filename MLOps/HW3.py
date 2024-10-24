from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
import mlflow
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import pandas as pd
from typing import Any, Dict, Literal, List
import logging
import io
from datetime import datetime, timedelta
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())



BUCKET = Variable.get("S3_BUCKET")

DEFAULT_ARGS = {
    "owner": "admoskalenko",
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}

FEATURES = ['account_length', 'phone_number',
       'international_plan', 'voice_mail_plan', 'number_vmail_messages',
       'total_day_minutes', 'total_day_calls', 'total_day_charge',
       'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
       'total_night_minutes', 'total_night_calls', 'total_night_charge',
       'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
       'number_customer_service_calls']

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

model_names = ["log_regression", "random_forest", "desicion_tree"]
models = dict(
    zip(model_names, [
        LogisticRegression(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
    ]))


def create_dag(dag_id: str, m_names: List, exp_name: str):

    ####### DAG STEPS #######

    def init() -> Dict[str, Any]:
        exps = mlflow.search_experiments(filter_string=f"name = '{exp_name}'")

        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _LOG.info(f'Pipeline started: {date}')

        if len(exps) > 0:
            experiment_id = exps[0].experiment_id
        else:
            experiment_id = mlflow.create_experiment(exp_name)
        mlflow.start_run(run_name="tau_ceti_pn", experiment_id = experiment_id, description = "parent")
        run = mlflow.active_run()
        metrics = dict()
        metrics['pipeline_start_date'] = date
        metrics['experiment_id'] = experiment_id
        metrics['run_id'] = run.info.run_id
        return metrics

    def get_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='init')

        metrics['download_date_begin'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        dataset = fetch_openml(name='churn', version=1)
        data = pd.concat([dataset["data"], pd.DataFrame(dataset["target"])], axis=1)

        metrics['download_date_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['dataset_shape'] = list(data.shape)

        s3_hook = S3Hook("s3_connection")
        filebuffer = io.BytesIO()
        data.to_pickle(filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"MoskalenkoAleksandr/datasets/churn.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
        _LOG.info('Data has been downloaded.')
        return metrics



    def prepare_data(**kwargs) -> Dict[str, Any]:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='get_data')
        metrics['prepared_date_begin'] = date

        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(
            key=f"MoskalenkoAleksandr/datasets/churn.pkl",
            bucket_name=BUCKET)
        df = pd.read_pickle(file)
        X, y = df[FEATURES], df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        scaler = StandardScaler()
        X_train_prepared = scaler.fit_transform(X_train)
        X_test_prepared = scaler.transform(X_test)
        X_train_prepared = pd.DataFrame(data=X_train_prepared, columns=list(scaler.get_feature_names_out()))
        X_test_prepared = pd.DataFrame(data=X_test_prepared, columns=list(scaler.get_feature_names_out()))

        for name, data in zip(
            ["X_train", "X_test", "y_train", "y_test"],
            [X_train_prepared, X_test_prepared, y_train, y_test],
        ):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"MoskalenkoAleksandr/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['prepared_date_end'] = date 
        metrics['features'] = FEATURES
        _LOG.info('Data has been prepared.')
        return metrics 




    def train_model(model_name, **kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids='prepare_data')
        data = {}
        s3_hook = S3Hook("s3_connection")
        for data_name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(key=f"MoskalenkoAleksandr/datasets/{data_name}.pkl", bucket_name=BUCKET)
            data[data_name] = pd.read_pickle(file)
        experiment_id = metrics['experiment_id']
        parent_run = metrics['run_id']
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, parent_run_id=parent_run, nested=True) as child_run:
            timestamps = {}
            timestamps['train_date_begin'] = date 
            model = models[model_name]
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            eval_df = X_test.copy()
            eval_df['labels'] = y_test.to_numpy()
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="labels",
                model_type="classifier",
                evaluators=["default"],
            )
            timestamps['train_date_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            _LOG.info('Model has been trained.')
            return timestamps


    def save_results(**kwargs) -> None:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids='prepare_data')
        for m_name in m_names:
            model_metrics = ti.xcom_pull(task_ids = f"train_{m_name}")
            metrics[f'{m_name}'] = model_metrics
        s3_hook = S3Hook("s3_connection")
        buff = io.BytesIO()
        buff.write(json.dumps(metrics, indent=2).encode())
        buff.seek(0)
        s3_hook.load_file_obj(file_obj = buff, key=f"MoskalenkoAleksandr/metrics/metrics.json", bucket_name=BUCKET, replace=True)

    ####### INIT DAG #######

    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        default_args=DEFAULT_ARGS
    )
    try:
        with dag:
            task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)
            task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag)
            task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag)
            tasks_train = [PythonOperator(task_id=f"train_{m_name}", python_callable=train_model, dag=dag, op_kwargs={'model_name': m_name}) for m_name in m_names]
            task_save_results = PythonOperator(task_id="save_result", python_callable=save_results, dag=dag)
            task_init >> task_get_data >> task_prepare_data >> tasks_train >> task_save_results
    finally:
        mlflow.end_run()

configure_mlflow()
create_dag(f"Aleksandr_Moskalenko", model_names, 'Aleksandr_Moskalenko')