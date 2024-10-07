from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
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
from typing import Any, Dict, Literal
import logging
import io
from datetime import datetime, timedelta
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

# YOUR IMPORTS HERE


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

model_names = ["log_regression", "random_forest", "desicion_tree"]
models = dict(
    zip(model_names, [
        LogisticRegression(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
    ]))


def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):

    ####### DAG STEPS #######

    def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _LOG.info(f'Pipeline started: {date}')
        metrics = dict()
        metrics['model_name'] = m_name 
        metrics['pipeline_start_date'] = date
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
            key=f"MoskalenkoAleksandr/{metrics['model_name']}/datasets/churn.pkl",
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
            key=f"MoskalenkoAleksandr/{metrics['model_name']}/datasets/churn.pkl",
            bucket_name=BUCKET)
        df = pd.read_pickle(file)
        X, y = df[FEATURES], df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        scaler = StandardScaler()
        X_train_prepared = scaler.fit_transform(X_train)
        X_test_prepared = scaler.transform(X_test)

        for name, data in zip(
            ["X_train", "X_test", "y_train", "y_test"],
            [X_train_prepared, X_test_prepared, y_train, y_test],
        ):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"MoskalenkoAleksandr/{metrics['model_name']}/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['prepared_date_end'] = date 
        metrics['features'] = FEATURES
        _LOG.info('Data has been prepared.')
        return metrics 




    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids='prepare_data')
        data = {}
        s3_hook = S3Hook("s3_connection")
        for data_name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(key=f"MoskalenkoAleksandr/{metrics['model_name']}/datasets/{data_name}.pkl", bucket_name=BUCKET)
            data[data_name] = pd.read_pickle(file)
            
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['train_date_begin'] = date 
        model = models[metrics['model_name']]
        X_train, y_train = data["X_train"], data["y_train"]
        model.fit(X_train, y_train)
        model_eval = {}
        X_test = data["X_test"]
        probas = model.predict_proba(X_test)[:, 1]
        y_predicted = (probas > 0.5).astype(int)
        y_true = data["y_test"]
        model_eval['roc_auc'] = roc_auc_score(y_true, probas)
        model_eval['f1_score'] = f1_score(y_true, y_predicted)
        model_eval['precision'] = precision_score(y_true, y_predicted)
        model_eval['recall'] = recall_score(y_true, y_predicted)
        metrics['metrics'] = model_eval
        metrics['train_date_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _LOG.info('Model has been trained.')
        return metrics


    def save_results(**kwargs) -> None:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids='train_model')
        s3_hook = S3Hook("s3_connection")
        buff = io.BytesIO()
        buff.write(json.dumps(metrics, indent=2).encode())
        buff.seek(0)
        s3_hook.load_file_obj(file_obj = buff, key=f"MoskalenkoAleksandr/{metrics['model_name']}/metrics/metrics.json", bucket_name=BUCKET, replace=True)

    ####### INIT DAG #######

    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        default_args=DEFAULT_ARGS
    )

    with dag:
        # YOUR TASKS HERE
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, op_kwargs={"m_name": m_name})

        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag)

        task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag)

        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)

        task_save_results = PythonOperator(task_id="save_result", python_callable=save_results, dag=dag)

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"Aleksandr_Moskalenko_{model_name}", model_name)