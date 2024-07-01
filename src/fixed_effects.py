import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import dagshub
import statsmodels.api as sm
import statsmodels.formula.api as smf

dagshub.init(repo_owner='Kajaani-Balabavan', repo_name='deep-panel', mlflow=True)
experiment_name = "Base Models"
mlflow.set_experiment(experiment_name)
model_name = "Fixed Effects Model"

# Load CSV data

# Transport Domain
file_path = r'..\data\processed\Passenger_Traffic_Los_Angeles.csv'
dataset = 'Passenger_Traffic_Los_Angeles'

# Environmental Domain
# file_path = r'..\data\processed\average-monthly-surface-temperature.csv'
# dataset = 'Average surface temperature_SAARC'

# Economic Domain
# file_path = r'..\data\processed\exchange_rate.csv'
# dataset = 'Exchange Rate per USD'

data = pd.read_csv(file_path)

# for surface temperature data
# data= data[['Entity','Day','Average surface temperature']]

# Rename columns
data.columns = ['Entity','Date', 'Value']
print(data.head())

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Handle outliers using Z-score method
# threshold = 3  # Adjust the threshold as needed
# z_scores = np.abs(zscore(data['Value']))
# data = data[(z_scores < threshold)]

# Encode entity IDs
entity_encoder = LabelEncoder()
data['Entity'] = entity_encoder.fit_transform(data['Entity'])

# Normalize the data using RobustScaler
scaler = MinMaxScaler()  # Use RobustScaler instead of MinMaxScaler
data['Value'] = scaler.fit_transform(data[['Value']])

# Split data into train and validation sets by entities
entities = data['Entity'].unique()
train_entities = entities[:int(0.8 * len(entities))]
val_entities = entities[int(0.8 * len(entities)):]

train_data = data[data['Entity'].isin(train_entities)]
val_data = data[data['Entity'].isin(val_entities)]

# Prepare the data for fixed effects model
train_data['Entity'] = train_data['Entity'].astype('category')
val_data['Entity'] = val_data['Entity'].astype('category')

# Define the fixed-effects model formula
formula = 'Value ~ Date + C(Entity)'

# Start MLflow tracking
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("scaling_method", "RobustScaler")
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("outlier_handling", "Z-score")
    # mlflow.log_param("outlier_threshold", threshold)

    # Fit the fixed-effects model
    model = smf.ols(formula, data=train_data).fit()

    # Log the model summary as an artifact
    with open("model_summary.txt", "w") as f:
        f.write(model.summary().as_text())
    mlflow.log_artifact("model_summary.txt")

    # Predict on train and validation data
    train_data['Predicted'] = model.predict(train_data)
    val_data['Predicted'] = model.predict(val_data)

    # Rescale predictions back to the original range
    train_data['Actual'] = scaler.inverse_transform(train_data[['Value']])
    train_data['Predicted'] = scaler.inverse_transform(train_data[['Predicted']])
    val_data['Actual'] = scaler.inverse_transform(val_data[['Value']])
    val_data['Predicted'] = scaler.inverse_transform(val_data[['Predicted']])

    # Evaluate Model
    def calculate_metrics(actuals, predictions):
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        return mae, mse, rmse, mape

    train_mae, train_mse, train_rmse, train_mape = calculate_metrics(train_data['Actual'], train_data['Predicted'])
    val_mae, val_mse, val_rmse, val_mape = calculate_metrics(val_data['Actual'], val_data['Predicted'])

    print(f'Train MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}')
    print(f'Validation MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}')

    # Log final metrics
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_mape", train_mape)

    mlflow.log_metric("val_mae", val_mae)
    mlflow.log_metric("val_mse", val_mse)
    mlflow.log_metric("val_rmse", val_rmse)
    mlflow.log_metric("val_mape", val_mape)

    # Plot train and validation predictions vs actuals
    plt.figure(figsize=(10, 5))
    plt.plot(train_data['Date'], train_data['Actual'], label='Train Actuals')
    plt.plot(train_data['Date'], train_data['Predicted'], label='Train Predictions')
    plt.legend()
    plt.title('Train Set: Predictions vs Actuals')
    plt.savefig('train_predictions_vs_actuals.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_data['Date'], val_data['Actual'], label='Validation Actuals')
    plt.plot(val_data['Date'], val_data['Predicted'], label='Validation Predictions')
    plt.legend()
    plt.title('Validation Set: Predictions vs Actuals')
    plt.savefig('val_predictions_vs_actuals.png')
    plt.close()

    # Log the plots as artifacts
    mlflow.log_artifact('train_predictions_vs_actuals.png')
    mlflow.log_artifact('val_predictions_vs_actuals.png')
