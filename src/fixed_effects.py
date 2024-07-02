# import mlflow
# from mlflow.models import infer_signature
# import pandas as pd
# import numpy as np
# from linearmodels import PanelOLS
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
# import matplotlib.pyplot as plt
# import dagshub
# import mlflow.pyfunc
# from scipy.stats import zscore

# def extract_time_features(df, date_column):
#     df[date_column] = pd.to_datetime(df[date_column])
#     df['Year'] = df[date_column].dt.year
#     df['Month'] = df[date_column].dt.month
#     df['Day'] = df[date_column].dt.day
#     df['Day of Week'] = df[date_column].dt.dayofweek
#     df['Week of Year'] = df[date_column].dt.isocalendar().week.astype(np.int32)
#     df['Quarter'] = df[date_column].dt.quarter
#     return df

# def create_lags(df, column, lags):
#     for lag in lags:
#         df[f'{column}_lag_{lag}'] = df[column].shift(lag).fillna(method='bfill')
#     return df

# def calculate_moving_average(df, column, window):
#     df[f'{column}_ma_{window}'] = df[column].rolling(window=window).mean().fillna(method='bfill')
#     return df

# dagshub.init(repo_owner='Kajaani-Balabavan', repo_name='deep-panel', mlflow=True)
# experiment_name = "Base Models"
# mlflow.set_experiment(experiment_name)
# model_name = "PanelOLS Model"

# # Load CSV data

# # Transport Domain
# file_path = r'..\data\processed\Passenger_Traffic_Los_Angeles.csv'
# dataset = 'Passenger_Traffic_Los_Angeles'

# # Environmental Domain
# # file_path = r'..\data\processed\average-monthly-surface-temperature.csv'
# # dataset = 'Average surface temperature_SAARC'

# # Economic Domain
# # file_path = r'..\data\processed\exchange_rate.csv'
# # dataset = 'Exchange Rate per USD'

# data = pd.read_csv(file_path)

# # for surface temperature data
# # data= data[['Entity','Day','Average surface temperature']]

# # Rename columns
# data.columns = ['Entity','Date', 'Value']
# print(data.head())

# # Convert 'Date' to datetime and set as index
# data['Date'] = pd.to_datetime(data['Date'])

# data = extract_time_features(data, 'Date')
# data = create_lags(data, 'Value', lags=[1, 2])
# data = calculate_moving_average(data, 'Value_lag_1', window=3)
# data = data.set_index('Date')

# print(data.head())
# print(data.columns)
# print(data.dtypes)

# # # Handle outliers using Z-score method
# # threshold = 3  # Adjust the threshold as needed
# # z_scores = np.abs(zscore(data['Value']))
# # data = data[(z_scores < threshold)]

# # Encode entity IDs
# entity_encoder = LabelEncoder()
# data['Entity'] = entity_encoder.fit_transform(data['Entity'])

# # Normalize the data using RobustScaler
# scaler = MinMaxScaler()  # Use MinMaxScaler
# data['Value'] = scaler.fit_transform(data[['Value']])

# # Split data into train and validation sets by entities
# entities = data['Entity'].unique()
# train_entities = entities[:int(0.8 * len(entities))]
# val_entities = entities[int(0.8 * len(entities)):]

# train_data = data[data['Entity'].isin(train_entities)]
# print(train_data.columns)
# val_data = data[data['Entity'].isin(val_entities)]

# # Prepare data for PanelOLS model
# # You'll need to specify the formula based on your features
# formula = 'Value ~ Year + Month + Day + Day of Week + Week of Year + Quarter + Value_lag_1 + Value_lag_2 + Value_lag_1_ma_3'
# # Fit the PanelOLS model
# model = PanelOLS.from_formula(formula, data=train_data)
# results = model.fit()

# # Make predictions
# train_predictions = results.predict(train_data)
# val_predictions = results.predict(val_data)

# # Evaluate Model
# def evaluate(model, loader):
#     model.eval()
#     actuals, predictions = [], []
#     with torch.no_grad():
#         for X_batch, y_batch, e_batch in loader:
#             y_pred = model(X_batch, e_batch)
#             # Rescale y_batch values back to the original range
#             actuals.extend(scaler.inverse_transform(y_batch.numpy().reshape(-1, 1)).flatten())
#             # Rescale predictions back to the original range
#             predictions.extend(scaler.inverse_transform(y_pred.numpy().reshape(-1, 1)).flatten())
#     return np.array(actuals), np.array(predictions)

# # train_actuals, train_predictions = evaluate(model, train_loader)
# # val_actuals, val_predictions = evaluate(model, val_loader)

# print('Training set size:', len(train_predictions))
# print('Validation set size:', len(val_predictions))
# print('Training set predictions:', len(train_predictions))
# print('Validation set predictions:', len(val_predictions))

# # Metrics
# def calculate_metrics(actuals, predictions):
#     mae = mean_absolute_error(actuals, predictions)
#     mse = mean_squared_error(actuals, predictions)
#     rmse = np.sqrt(mse)
#     mape = mean_absolute_percentage_error(actuals, predictions) * 100
#     return mae, mse, rmse, mape

# train_mae, train_mse, train_rmse, train_mape = calculate_metrics(train_data['Value'], train_predictions)
# val_mae, val_mse, val_rmse, val_mape = calculate_metrics(val_data['Value'], val_predictions)

# print(f'Train MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}')
# print(f'Validation MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}')

# # Log results to MLflow
# with mlflow.start_run():
#     # Log hyperparameters
#     mlflow.log_param("dataset", dataset)
#     mlflow.log_param("model_name", model_name)
#     mlflow.log_param("outlier_handling", "Z-score")
#     # mlflow.log_param("outlier_threshold", threshold)
#     mlflow.log_param("scaling_method", "RobustScaler")

#     # Log model parameters
#     # You can log model parameters if you have any specific ones you'd like to track.

#     # Evaluate model and log metrics
#     mlflow.log_metric("train_mae", train_mae)
#     mlflow.log_metric("train_mse", train_mse)
#     mlflow.log_metric("train_rmse", train_rmse)
#     mlflow.log_metric("train_mape", train_mape)

#     mlflow.log_metric("val_mae", val_mae)
#     mlflow.log_metric("val_mse", val_mse)
#     mlflow.log_metric("val_rmse", val_rmse)
#     mlflow.log_metric("val_mape", val_mape)

#     plt.figure(figsize=(10, 5))
#     plt.plot(train_actuals, label='Train Actuals')
#     plt.plot(train_predictions, label='Train Predictions')
#     plt.legend()
#     plt.title('Train Set: Predictions vs Actuals')
#     plt.savefig('train_predictions_vs_actuals.png')
#     plt.close()

#     plt.figure(figsize=(10, 5))
#     plt.plot(val_actuals, label='Validation Actuals')
#     plt.plot(val_predictions, label='Validation Predictions')
#     plt.legend()
#     plt.title('Validation Set: Predictions vs Actuals')
#     plt.savefig('val_predictions_vs_actuals.png')
#     plt.close()

#     # Log the plots as artifacts
#     mlflow.log_artifact('train_predictions_vs_actuals.png')
#     mlflow.log_artifact('val_predictions_vs_actuals.png')
    
#     # Save model
#     mlflow.pytorch.log_model(model, "best_model")

#     # Define a custom Python model class for the scaler
#     class ScalerWrapper(mlflow.pyfunc.PythonModel):
#         def __init__(self, scaler):
#             self.scaler = scaler

#         def predict(self, context, model_input):
#             return self.scaler.transform(model_input)

#     # Wrap the scaler
#     scaler_wrapper = ScalerWrapper(scaler)

#     # Log the scaler as a custom Python model
#     mlflow.pyfunc.log_model("scaler", python_model=scaler_wrapper)


import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import dagshub
from linearmodels.panel import RandomEffects

def extract_time_features(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    df['Day of Week'] = df[date_column].dt.dayofweek
    df['Week of Year'] = df[date_column].dt.isocalendar().week.astype(np.int32)
    df['Quarter'] = df[date_column].dt.quarter
    return df

def create_lags(df, column, lags):
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag).fillna(method='bfill')
    return df

def calculate_moving_average(df, column, window):
    df[f'{column}_ma_{window}'] = df[column].rolling(window=window).mean().fillna(method='bfill')
    return df

dagshub.init(repo_owner='Kajaani-Balabavan', repo_name='deep-panel', mlflow=True)
experiment_name = "Base Models"
mlflow.set_experiment(experiment_name)
model_name = "Random Effects"

# Load CSV data

# Transport Domain
# file_path = r'..\data\processed\Passenger_Traffic_Los_Angeles.csv'
# dataset = 'Passenger_Traffic_Los_Angeles'

# Environmental Domain
file_path = r'..\data\processed\average-monthly-surface-temperature.csv'
dataset = 'Average surface temperature_SAARC'

# Economic Domain
# file_path = r'..\data\processed\exchange_rate.csv'
# dataset = 'Exchange Rate per USD'

data = pd.read_csv(file_path)

# for surface temperature data
data= data[['Entity','Day','Average surface temperature']]

# Rename columns
data.columns = ['Entity','Date', 'Value']
print(data.head())

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])

data = extract_time_features(data, 'Date')
data = create_lags(data, 'Value', lags=[1, 12])  # fill with lags you want. 1 is compulsory, you can put as many lags you want in the list. forexample, if daily data you may want lag 7
data = calculate_moving_average(data, 'Value_lag_1', window=4) #set the window value to no of data points you want to take average of
data = data.set_index('Date')

print(data.head())
print(data.columns)
print(data.dtypes)

# Encode entity IDs
entity_encoder = LabelEncoder()
data['Entity'] = entity_encoder.fit_transform(data['Entity'])

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()  # Use MinMaxScaler
data['Value'] = scaler.fit_transform(data[['Value']])

# Split data into train and validation sets by entities
entities = data['Entity'].unique()
train_entities = entities[:int(0.8 * len(entities))]
val_entities = entities[int(0.2 * len(entities)):]

train_data = data[data['Entity'].isin(train_entities)]
val_data = data[data['Entity'].isin(val_entities)]

# Prepare data for Random Effects model
train_data_re = train_data.reset_index()
val_data_re = val_data.reset_index()

exog_vars = ['Year', 'Month', 'Day of Week', 'Week of Year', 'Quarter', 'Value_lag_1', 'Value_lag_12', 'Value_lag_1_ma_4']
exog = train_data_re[exog_vars]
endog = train_data_re['Value']

# Start MLflow tracking
with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("outlier_handling", "Z-score")
    mlflow.log_param("scaling_method", "MinMaxScaler")

    # Fit the Random Effects model
    re_model = RandomEffects(endog, exog)
    re_results = re_model.fit()
    print(re_results)

    # Log the model summary
    with open("model_summary.txt", "w") as f:
        f.write(re_results.summary.as_text())
    mlflow.log_artifact("model_summary.txt")

    # Predict on train and validation data
    train_predictions = re_results.predict(exog)
    val_predictions = re_results.predict(val_data_re[exog_vars])

    # Metrics
    def calculate_metrics(actuals, predictions):
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        r2 = r2_score(actuals, predictions)
        return mae, mse, rmse, mape, r2

    train_mae, train_mse, train_rmse, train_mape, train_r2 = calculate_metrics(train_data_re['Value'], train_predictions)
    val_mae, val_mse, val_rmse, val_mape, val_r2 = calculate_metrics(val_data_re['Value'], val_predictions)

    print(f'Train MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}, R2: {train_r2:.4f}')
    print(f'Validation MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}, R2: {val_r2:.4f}')

    # Log final metrics
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_mape", train_mape)
    mlflow.log_metric("train_r2", train_r2)

    mlflow.log_metric("val_mae", val_mae)
    mlflow.log_metric("val_mse", val_mse)
    mlflow.log_metric("val_rmse", val_rmse)
    mlflow.log_metric("val_mape", val_mape)
    mlflow.log_metric("val_r2", val_r2)

    plt.figure(figsize=(10, 5))
    plt.plot(train_data_re.index, train_data_re['Value'], label='Train Actuals')
    plt.plot(train_data_re.index, train_predictions, label='Train Predictions')
    plt.legend()
    plt.title('Train Set: Predictions vs Actuals')
    plt.savefig('train_predictions_vs_actuals.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_data_re.index, val_data_re['Value'], label='Validation Actuals')
    plt.plot(val_data_re.index, val_predictions, label='Validation Predictions')
    plt.legend()
    plt.title('Validation Set: Predictions vs Actuals')
    plt.savefig('val_predictions_vs_actuals.png')
    plt.close()

    # Log the plots as artifacts
    mlflow.log_artifact('train_predictions_vs_actuals.png')
    mlflow.log_artifact('val_predictions_vs_actuals.png')

    # Log model parameters
    mlflow.log_param("params", re_results.params.to_string())

    # Save and log the model
    mlflow.log_model(re_results, artifact_path="model")

print('Experiment is logged to MLflow')
