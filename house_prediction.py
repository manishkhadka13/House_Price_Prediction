import mlflow
from mlflow import log_metric, log_param, log_artifact
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,classification_report
from sklearn.pipeline import make_pipeline
import pickle
import datetime

# Set the experiment name in MLflow
mlflow.set_experiment('House Prediction')

data = pd.read_csv('data/cleaned_data.csv')

X = data.drop(columns=['price'])
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run(run_name=f'Regression for house price {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'):
    # Creating a column transformer to one hot encode the location column and scale the other columns
    column_trans = make_column_transformer((OneHotEncoder(sparse=False), ['location']), remainder='passthrough')

    sclr = StandardScaler()
    lnr = LinearRegression()
    pipe = make_pipeline(column_trans, sclr, lnr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Save the trained model using pickle
    model_path = './models/prediction.pkl'
    with open(model_path, 'wb') as model_file:
        pickle.dump(pipe, model_file)

    # Logging information to MLflow
    log_param('Model', 'Linear Regression')
    log_param('Scaler', 'StandardScaler')
    log_metric('Mean Squared Error', mean_squared_error(y_test, y_pred))
    log_metric('R2 Score', pipe.score(X_test, y_test))
    mlflow.log_artifact(model_path)

