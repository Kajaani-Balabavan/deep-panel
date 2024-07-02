import mlflow
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import dagshub

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
        df[f'{column}lag{lag}'] = df[column].shift(lag).fillna(method='bfill')
    return df

def calculate_moving_average(df, column, window):
    df[f'{column}ma{window}'] = df[column].rolling(window=window).mean().fillna(method='bfill')
    return df

dagshub.init(repo_owner='Kajaani-Balabavan', repo_name='deep-panel', mlflow=True)
experiment_name = "Base Models"
mlflow.set_experiment(experiment_name)
model_name = "Pooled OLS"

# Load CSV data
file_path = r'..\data\processed\Passenger_Traffic_Los_Angeles.csv'
dataset = 'Passenger_Traffic_Los_Angeles'

# Environmental Domain
# file_path = r'..\data\processed\average-monthly-surface-temperature.csv'
# dataset = 'Average surface temperature_SAARC'

# Economic Domain
# file_path = r'..\data\processed\exchange_rate.csv'
# dataset = 'Exchange Rate per USD'

# Covid_deaths
# file_path = r'..\data\Base_Paper\Predictions for COVID-19 with deep learning models of LSTM, GRU, and Bi-LSTM\death_final.csv'
# dataset = 'Covid Deaths'

# Covid_confirmed
# file_path = r'..\data\Base_Paper\Predictions for COVID-19 with deep learning models of LSTM, GRU, and Bi-LSTM\confirmed_final.csv'
# dataset = 'Covid Confirmed'

# Covid_recovered
# file_path = r'..\data\Base_Paper\Predictions for COVID-19 with deep learning models of LSTM, GRU, and Bi-LSTM\recovered_final.csv'
# dataset = 'Covid Recovered'

# Africa GDP
# file_path = r'..\data\Base_Paper\African GDP\Final_African_GDP.csv'
# dataset = 'GDP Africa'

data = pd.read_csv(file_path)

# Rename columns
data.columns = ['Entity','Date', 'Value']

# Affrica GDP
# data.columns = ['Entity','Date', 'Value','GNI','PPP']
print(data.head())

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])

data = extract_time_features(data, 'Date')
data = create_lags(data, 'Value', lags=[1, 7])
data = calculate_moving_average(data, 'Valuelag1', window=4)
data = data.set_index('Date')

print(data.head())
print(data.columns)
print(data.dtypes)

# Encode entity IDs
entity_encoder = LabelEncoder()
data['Entity'] = entity_encoder.fit_transform(data['Entity'])

# Normalize the data using Minmaxscaler
scaler = MinMaxScaler()
data['Value'] = scaler.fit_transform(data[['Value']])

# Split the data into training and validation sets
entities = data['Entity'].unique()
train_data = pd.DataFrame()
val_data = pd.DataFrame()
for entity in entities:
    entity_data = data[data['Entity'] == entity]
    split_index = int(0.8 * len(entity_data))
    entity_train_data = entity_data[:split_index]
    train_data = train_data._append(entity_train_data)
    entity_val_data = entity_data[split_index:]
    val_data = val_data._append(entity_val_data)
print("train data \n", train_data.head())
print("validation data \n", val_data.head())

# Plotting the actual values for each entity with different colors
plt.figure(figsize=(10, 5))
for entity in entities:
    entity_data = data[data['Entity'] == entity]
    plt.plot(entity_data.index, entity_data['Value'], label=f'Entity {entity}')
plt.legend()
plt.title('Actual Values for Each Entity')
plt.savefig('actual_values_per_entity.png')
plt.close()

# Log the plot as an artifact before training
with mlflow.start_run():
    mlflow.log_artifact('actual_values_per_entity.png')

# Train Pooled OLS Model
formula = 'Value ~ Year + Month + Q("Day of Week") + Q("Week of Year") + Quarter + Valuelag1 + Valuelag7 + Valuelag1ma4'

# Affrica GDP
# formula = 'Value ~Year + Valuelag1 + GNI+PPP'
model = smf.ols(formula, train_data).fit()

# Start MLflow tracking
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("formula", formula)

    # Log model summary
    mlflow.log_text(str(model.summary()), "model_summary.txt")

    # Make predictions on train and validation sets
    train_predictions = model.predict(train_data)
    val_predictions = model.predict(val_data)

    # Inverse transform the predictions
    train_data['Value'] = scaler.inverse_transform(train_data[['Value']])
    train_predictions = scaler.inverse_transform(train_predictions.values.reshape(-1, 1)).flatten()
    val_data['Value'] = scaler.inverse_transform(val_data[['Value']])
    val_predictions = scaler.inverse_transform(val_predictions.values.reshape(-1, 1)).flatten()

    # Metrics
    def calculate_metrics(actuals, predictions):
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        r2 = r2_score(actuals, predictions)
        return mae, mse, rmse, mape, r2

    train_mae, train_mse, train_rmse, train_mape, train_r2 = calculate_metrics(train_data['Value'], train_predictions)
    val_mae, val_mse, val_rmse, val_mape, val_r2 = calculate_metrics(val_data['Value'], val_predictions)

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

    # Plotting the results with different colors for each entity
    plt.figure(figsize=(10, 5))
    for entity in entities:
        entity_train_data = train_data[train_data['Entity'] == entity]
        plt.plot(entity_train_data.index, entity_train_data['Value'], label=f'Train Actuals Entity {entity}', alpha=0.6)
        plt.plot(entity_train_data.index, train_predictions[train_data['Entity'] == entity], label=f'Train Predictions Entity {entity}', alpha=0.6)
    plt.legend()
    plt.title('Train Set: Predictions vs Actuals')
    plt.savefig('train_predictions_vs_actuals.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for entity in entities:
        entity_val_data = val_data[val_data['Entity'] == entity]
        plt.plot(entity_val_data.index, entity_val_data['Value'], label=f'Validation Actuals Entity {entity}', alpha=0.6)
        plt.plot(entity_val_data.index, val_predictions[val_data['Entity'] == entity], label=f'Validation Predictions Entity {entity}', alpha=0.6)
    plt.legend()
    plt.title('Validation Set: Predictions vs Actuals')
    plt.savefig('val_predictions_vs_actuals.png')
    plt.close()

    # Log the plots as artifacts
    mlflow.log_artifact('train_predictions_vs_actuals.png')
    mlflow.log_artifact('val_predictions_vs_actuals.png')

    # Log the model
    model.save('pooled_ols_model.pkl')
    mlflow.log_artifact('pooled_ols_model.pkl')

print("Pooled OLS Model training and evaluation complete.")
