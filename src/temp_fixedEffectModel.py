# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from linearmodels.panel import PanelOLS

# # Economic Domain
# file_path = 'exchange_rate_without_singapore.csv'
# dataset = 'Exchange Rate per USD'

# # Load CSV data
# data = pd.read_csv(r'..\data\processed\exchange_rate.csv')
# # print(data.head())

# # Rename columns
# data.columns = ['Entity','Date', 'Value']
# # print(data.head())

# # Convert 'Date' to datetime
# data['Date'] = pd.to_datetime(data['Date'])
# # print(data.head())



# # DATA PREPROCESSING 
# # Encode entity IDs
# entity_encoder = LabelEncoder()
# data['Entity'] = entity_encoder.fit_transform(data['Entity'])
# # print(data.head())
# # Normalize the data 
# scaler = MinMaxScaler()  # scales each feature to the [0, 1] range.
# data['Value'] = scaler.fit_transform(data[['Value']])
# # print(data.head())

# # Split data into train and validation sets by entities
# entities = data['Entity'].unique()
# train_data=pd.DataFrame()
# val_data=pd.DataFrame()
# for entity in entities:
#     entity_data = data[data['Entity'] == entity]
#     split_index = int(0.8 * len(entity_data))
#     entity_train_data = entity_data[:split_index]
#     train_data = train_data._append(entity_train_data,ignore_index=True)
#     entity_val_data = entity_data[split_index:]
#     val_data = val_data._append(entity_val_data,ignore_index=True)
# # print("train data \n",train_data.head())
# # print("validation data \n", val_data.head())

# # plotting the train-validation data split
# # plt.figure(figsize=(14,8))
# # for entity, group in train_data.groupby('Entity'):
# #     plt.plot(group['Date'],group['Value'],label=f'Entity-train {entity}')
# # for entity, group in val_data.groupby('Entity'):
# #     plt.plot(group['Date'],group['Value'],label=f'Entity-val {entity}')
# # plt.title('Exchange Rate train, validation split')
# # plt.xlabel('Date')
# # plt.ylabel('Normalized exchange rate')
# # plt.legend(title='Entity')
# # plt.grid(True)
# # plt.savefig('train-validation split.png')

# # Prepare the data for fixed effects model
# train_data['Entity'] = train_data['Entity'].astype('category')
# val_data['Entity'] = val_data['Entity'].astype('category')
# # print("train data \n",train_data.head())
# # print("validation data \n", val_data.head())

# # Set multi-index (Entity and Date) for panel data structure
# train_data = train_data.set_index(['Entity', 'Date'])
# # Add a constant term to the model (intercept)
# train_data['Intercept'] = 1
# # Specify the dependent variable (y) and independent variable (x)
# y = train_data['Value']
# x = train_data[['Intercept']]  # Add other independent variables if necessary
# # print("train data \n",train_data.head())
# # Fit the fixed effects model
# # model = PanelOLS(y, x, entity_effects=True)
# # results = model.fit()

# model = PanelOLS(y, x, entity_effects=True)  # Explicitly set observed
# results = model.fit()

# # Print the model summary
# # print(results.summary)


# # Extract fitted values and residuals
# fitted_values = results.fitted_values
# residuals = results.resids
# # Plot actual vs. fitted values
# plt.figure(figsize=(14, 8))
# for entity in train_data.index.get_level_values('Entity').unique():
#     entity_data = train_data.loc[entity]
#     entity_fitted_values = fitted_values.loc[entity]
#     plt.plot(entity_data.index.get_level_values('Date'), entity_data['Value'], label=f'Actual {entity}')
#     plt.plot(entity_fitted_values.index.get_level_values('Date'), entity_fitted_values, linestyle='--', label=f'Fitted {entity}')
    

# plt.title('Actual vs. Fitted Values')
# plt.xlabel('Date')
# plt.ylabel('Normalized Exchange Rate')
# plt.legend()
# plt.grid(True)
# plt.savefig('line_plot_actual_vs_fitted.png')

# # Plot residuals as a line plot
# plt.figure(figsize=(14, 8))
# for entity in train_data.index.get_level_values('Entity').unique():
#     entity_residuals = residuals.loc[entity]
#     plt.plot(entity_residuals.index.get_level_values('Date'), entity_residuals, label=f'Entity {entity}')


# plt.title('Residuals Over Time')
# plt.xlabel('Date')
# plt.ylabel('Residuals')
# plt.axhline(0, color='red', linestyle='--', linewidth=1)
# plt.legend()
# plt.grid(True)
# plt.savefig('line_plot_residuals.png')

# # Plotting actual vs. fitted values
# # plt.figure(figsize=(14, 8))
# # plt.scatter(train_data.index.get_level_values('Date'), y, label='Actual Values', alpha=0.6)
# # plt.scatter(train_data.index.get_level_values('Date'), fitted_values, label='Fitted Values', alpha=0.6)
# # plt.title('Actual vs. Fitted Values')
# # plt.xlabel('Date')
# # plt.ylabel('Normalized Exchange Rate')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('scatter_plot_actual_vs_fitted.png')

# # # Plotting residuals
# # plt.figure(figsize=(14, 8))
# # plt.scatter(train_data.index.get_level_values('Date'), residuals, label='Residuals', alpha=0.6)
# # plt.title('Residuals')
# # plt.xlabel('Date')
# # plt.ylabel('Residuals')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('scatter_plot_residuals.png')

# # Plotting the graph for each entity
# # plt.figure(figsize=(14, 8))
# # for entity, group in data.groupby('Entity'):
# #     plt.plot(group['Date'], group['Value'], label=f'Entity {entity}')

# # plt.title('Normalized Exchange Rate per USD over Time')
# # plt.xlabel('Date')
# # plt.ylabel('Normalized Exchange Rate')
# # plt.legend(title='Entity')
# # plt.grid(True)
# # # plt.show()
# # plt.savefig('exchange_rate_plot_after_normalized.png')

# import mlflow
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
# import matplotlib.pyplot as plt
# import dagshub

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
# model_name = "Random Effects"

# # Load CSV data
# file_path = r'..\data\processed\average-monthly-surface-temperature.csv'
# dataset = 'Average surface temperature_SAARC'

# data = pd.read_csv(file_path)
# data = data[['Entity','Day','Average surface temperature']]

# # Rename columns
# data.columns = ['Entity','Date', 'Value']
# print(data.head())

# # Convert 'Date' to datetime and set as index
# data['Date'] = pd.to_datetime(data['Date'])

# data = extract_time_features(data, 'Date')
# data = create_lags(data, 'Value', lags=[1, 12])
# data = calculate_moving_average(data, 'Value_lag_1', window=4)
# data = data.set_index('Date')

# print(data.head())
# print(data.columns)
# print(data.dtypes)

# # Encode entity IDs
# entity_encoder = LabelEncoder()
# data['Entity'] = entity_encoder.fit_transform(data['Entity'])

# # Normalize the data using Minmaxscaler
# scaler = MinMaxScaler()
# data['Value'] = scaler.fit_transform(data[['Value']])

# # Split the data into training and validation sets
# entities = data['Entity'].unique()
# train_data = pd.DataFrame()
# val_data = pd.DataFrame()
# for entity in entities:
#     entity_data = data[data['Entity'] == entity]
#     split_index = int(0.8 * len(entity_data))
#     entity_train_data = entity_data[:split_index]
#     train_data = train_data._append(entity_train_data)
#     entity_val_data = entity_data[split_index:]
#     val_data = val_data._append(entity_val_data)
# print("train data \n", train_data.head())
# print("validation data \n", val_data.head())

# # Train Random Effects Model
# formula = 'Value ~ Year + Month + Q("Day of Week") + Q("Week of Year") + Quarter + Value_lag_1 + Value_lag_12 + Value_lag_1_ma_4'
# model = smf.mixedlm(formula, train_data, groups=train_data["Entity"]).fit()

# # Start MLflow tracking
# with mlflow.start_run():
#     # Log hyperparameters
#     mlflow.log_param("model_name", model_name)
#     mlflow.log_param("formula", formula)

#     # Log model summary
#     mlflow.log_text(str(model.summary()), "model_summary.txt")

#     # Make predictions on train and validation sets
#     train_predictions = model.predict(train_data)
#     val_predictions = model.predict(val_data)

#     # Inverse transform the predictions
#     train_data['Value'] = scaler.inverse_transform(train_data[['Value']])
#     train_predictions = scaler.inverse_transform(train_predictions.values.reshape(-1, 1)).flatten()
#     val_data['Value'] = scaler.inverse_transform(val_data[['Value']])
#     val_predictions = scaler.inverse_transform(val_predictions.values.reshape(-1, 1)).flatten()

#     # Metrics
#     def calculate_metrics(actuals, predictions):
#         mae = mean_absolute_error(actuals, predictions)
#         mse = mean_squared_error(actuals, predictions)
#         rmse = np.sqrt(mse)
#         mape = mean_absolute_percentage_error(actuals, predictions) * 100
#         r2 = r2_score(actuals, predictions)
#         return mae, mse, rmse, mape, r2

#     train_mae, train_mse, train_rmse, train_mape, train_r2 = calculate_metrics(train_data['Value'], train_predictions)
#     val_mae, val_mse, val_rmse, val_mape, val_r2 = calculate_metrics(val_data['Value'], val_predictions)

#     print(f'Train MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}, R2: {train_r2:.4f}')
#     print(f'Validation MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}, R2: {val_r2:.4f}')

#     # Log final metrics
#     mlflow.log_metric("train_mae", train_mae)
#     mlflow.log_metric("train_mse", train_mse)
#     mlflow.log_metric("train_rmse", train_rmse)
#     mlflow.log_metric("train_mape", train_mape)
#     mlflow.log_metric("train_r2", train_r2)

#     mlflow.log_metric("val_mae", val_mae)
#     mlflow.log_metric("val_mse", val_mse)
#     mlflow.log_metric("val_rmse", val_rmse)
#     mlflow.log_metric("val_mape", val_mape)
#     mlflow.log_metric("val_r2", val_r2)

#     # Plotting the results
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_data.index, train_data['Value'], label='Train Actuals')
#     plt.plot(train_data.index, train_predictions, label='Train Predictions')
#     plt.legend()
#     plt.title('Train Set: Predictions vs Actuals')
#     plt.savefig('train_predictions_vs_actuals.png')
#     plt.close()

#     plt.figure(figsize=(10, 5))
#     plt.plot(val_data.index, val_data['Value'], label='Validation Actuals')
#     plt.plot(val_data.index, val_predictions, label='Validation Predictions')
#     plt.legend()
#     plt.title('Validation Set: Predictions vs Actuals')
#     plt.savefig('val_predictions_vs_actuals.png')
#     plt.close()

#     # Log the plots as artifacts
#     mlflow.log_artifact('train_predictions_vs_actuals.png')
#     mlflow.log_artifact('val_predictions_vs_actuals.png')

#     # Log the model
#     model.save('random_effects_model.pkl')
#     mlflow.log_artifact('random_effects_model.pkl')

# print("Random Effects Model training and evaluation complete.")



# import mlflow
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
# import matplotlib.pyplot as plt
# import dagshub

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
# model_name = "Random Effects"

# # Load CSV data
# file_path = r'..\data\processed\average-monthly-surface-temperature.csv'
# dataset = 'Average surface temperature_SAARC'

# data = pd.read_csv(file_path)
# data = data[['Entity','Day','Average surface temperature']]

# # Rename columns
# data.columns = ['Entity','Date', 'Value']
# print(data.head())

# # Convert 'Date' to datetime and set as index
# data['Date'] = pd.to_datetime(data['Date'])

# data = extract_time_features(data, 'Date')
# data = create_lags(data, 'Value', lags=[1, 12])
# data = calculate_moving_average(data, 'Value_lag_1', window=4)
# data = data.set_index('Date')

# print(data.head())
# print(data.columns)
# print(data.dtypes)

# # Encode entity IDs
# entity_encoder = LabelEncoder()
# data['Entity'] = entity_encoder.fit_transform(data['Entity'])

# # Normalize the data using Minmaxscaler
# scaler = MinMaxScaler()
# data['Value'] = scaler.fit_transform(data[['Value']])

# # Split the data into training and validation sets
# entities = data['Entity'].unique()
# train_data = pd.DataFrame()
# val_data = pd.DataFrame()
# for entity in entities:
#     entity_data = data[data['Entity'] == entity]
#     split_index = int(0.8 * len(entity_data))
#     entity_train_data = entity_data[:split_index]
#     train_data = train_data._append(entity_train_data)
#     entity_val_data = entity_data[split_index:]
#     val_data = val_data._append(entity_val_data)
# print("train data \n", train_data.head())
# print("validation data \n", val_data.head())

# # Train Random Effects Model
# formula = 'Value ~ Year + Month + Q("Day of Week") + Q("Week of Year") + Quarter + Value_lag_1 + Value_lag_12 + Value_lag_1_ma_4'
# model = smf.mixedlm(formula, train_data, groups=train_data["Entity"]).fit()

# # Start MLflow tracking
# with mlflow.start_run():
#     # Log hyperparameters
#     mlflow.log_param("model_name", model_name)
#     mlflow.log_param("formula", formula)

#     # Log model summary
#     mlflow.log_text(str(model.summary()), "model_summary.txt")

#     # Make predictions on train and validation sets
#     train_predictions = model.predict(train_data)
#     val_predictions = model.predict(val_data)

#     # Inverse transform the predictions
#     train_data['Value'] = scaler.inverse_transform(train_data[['Value']])
#     train_predictions = scaler.inverse_transform(train_predictions.values.reshape(-1, 1)).flatten()
#     val_data['Value'] = scaler.inverse_transform(val_data[['Value']])
#     val_predictions = scaler.inverse_transform(val_predictions.values.reshape(-1, 1)).flatten()

#     # Metrics
#     def calculate_metrics(actuals, predictions):
#         mae = mean_absolute_error(actuals, predictions)
#         mse = mean_squared_error(actuals, predictions)
#         rmse = np.sqrt(mse)
#         mape = mean_absolute_percentage_error(actuals, predictions) * 100
#         r2 = r2_score(actuals, predictions)
#         return mae, mse, rmse, mape, r2

#     train_mae, train_mse, train_rmse, train_mape, train_r2 = calculate_metrics(train_data['Value'], train_predictions)
#     val_mae, val_mse, val_rmse, val_mape, val_r2 = calculate_metrics(val_data['Value'], val_predictions)

#     print(f'Train MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}, R2: {train_r2:.4f}')
#     print(f'Validation MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}, R2: {val_r2:.4f}')

#     # Log final metrics
#     mlflow.log_metric("train_mae", train_mae)
#     mlflow.log_metric("train_mse", train_mse)
#     mlflow.log_metric("train_rmse", train_rmse)
#     mlflow.log_metric("train_mape", train_mape)
#     mlflow.log_metric("train_r2", train_r2)

#     mlflow.log_metric("val_mae", val_mae)
#     mlflow.log_metric("val_mse", val_mse)
#     mlflow.log_metric("val_rmse", val_rmse)
#     mlflow.log_metric("val_mape", val_mape)
#     mlflow.log_metric("val_r2", val_r2)

#     # Plotting the results with different colors for each entity
#     plt.figure(figsize=(10, 5))
#     for entity in entities:
#         entity_train_data = train_data[train_data['Entity'] == entity]
#         plt.plot(entity_train_data.index, entity_train_data['Value'], label=f'Train Actuals Entity {entity}', alpha=0.6)
#         plt.plot(entity_train_data.index, train_predictions[train_data['Entity'] == entity], label=f'Train Predictions Entity {entity}', alpha=0.6)
#     plt.legend()
#     plt.title('Train Set: Predictions vs Actuals')
#     plt.savefig('train_predictions_vs_actuals.png')
#     plt.close()

#     plt.figure(figsize=(10, 5))
#     for entity in entities:
#         entity_val_data = val_data[val_data['Entity'] == entity]
#         plt.plot(entity_val_data.index, entity_val_data['Value'], label=f'Validation Actuals Entity {entity}', alpha=0.6)
#         plt.plot(entity_val_data.index, val_predictions[val_data['Entity'] == entity], label=f'Validation Predictions Entity {entity}', alpha=0.6)
#     plt.legend()
#     plt.title('Validation Set: Predictions vs Actuals')
#     plt.savefig('val_predictions_vs_actuals.png')
#     plt.close()

#     # Log the plots as artifacts
#     mlflow.log_artifact('train_predictions_vs_actuals.png')
#     mlflow.log_artifact('val_predictions_vs_actuals.png')

#     # Log the model
#     model.save('random_effects_model.pkl')
#     mlflow.log_artifact('random_effects_model.pkl')

# print("Random Effects Model training and evaluation complete.")

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
file_path = r'..\data\processed\average-monthly-surface-temperature.csv'
dataset = 'Average surface temperature_SAARC'

data = pd.read_csv(file_path)
data = data[['Entity','Day','Average surface temperature']]

# Rename columns
data.columns = ['Entity','Date', 'Value']
print(data.head())

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])

data = extract_time_features(data, 'Date')
data = create_lags(data, 'Value', lags=[1, 12])
data = calculate_moving_average(data, 'Value_lag_1', window=4)
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

# Train Random Effects Model
formula = 'Value ~ Year + Month + Q("Day of Week") + Q("Week of Year") + Quarter + Value_lag_1 + Value_lag_12 + Value_lag_1_ma_4'
model = smf.mixedlm(formula, train_data, groups=train_data["Entity"]).fit()

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
    model.save('random_effects_model.pkl')
    mlflow.log_artifact('random_effects_model.pkl')

print("Random Effects Model training and evaluation complete.")
