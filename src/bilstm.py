import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import dagshub
import mlflow.pyfunc
from scipy.stats import zscore

dagshub.init(repo_owner='Kajaani-Balabavan', repo_name='deep-panel', mlflow=True)
experiment_name = "Base Models"
mlflow.set_experiment(experiment_name)
model_name = "Bi-LSTM"

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

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# # Handle outliers using Z-score method
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

# Function to create sequences with entity embeddings
def create_sequences(data, seq_length):
    xs, ys, es = [], [], []
    for entity in data['Entity'].unique():
        entity_data = data[data['Entity'] == entity]
        for i in range(len(entity_data) - seq_length):
            x = entity_data['Value'].values[i:i+seq_length]
            y = entity_data['Value'].values[i+seq_length]
            e = entity_data['Entity'].values[i:i+seq_length]
            xs.append(x)
            ys.append(y)
            es.append(e)
    return np.array(xs), np.array(ys), np.array(es)

# Create sequences
seq_length = 10
X_train, y_train, e_train = create_sequences(train_data, seq_length)
X_val, y_val, e_val = create_sequences(val_data, seq_length)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float().unsqueeze(-1)
y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
e_train = torch.from_numpy(e_train).long()

X_val = torch.from_numpy(X_val).float().unsqueeze(-1)
y_val = torch.from_numpy(y_val).float().unsqueeze(-1)
e_val = torch.from_numpy(e_val).long()

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train, e_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, y_val, e_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define BiLSTM Model with Entity Embeddings
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout_prob, num_entities, embedding_dim):
        super(BiLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.embedding = nn.Embedding(num_entities, embedding_dim)
        self.lstm = nn.LSTM(input_size + embedding_dim, hidden_layer_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_seq, entity_seq):
        entity_embedding = self.embedding(entity_seq).squeeze()
        input_combined = torch.cat((input_seq, entity_embedding), dim=-1)
        lstm_out, _ = self.lstm(input_combined)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Hyperparameters
input_size = 1
hidden_layer_size = 64
output_size = 1
dropout_prob = 0.2
learning_rate = 0.001
embedding_dim = 5  # Example embedding dimension
num_entities = len(entities)

# Initialize the model, loss function, and optimizer
model = BiLSTM(input_size, hidden_layer_size, output_size, dropout_prob, num_entities, embedding_dim)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training with early stopping
epochs = 300
patience = 10
best_val_loss = float('inf')
patience_counter = 0

# Start MLflow tracking
with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param("input_size", input_size)
    mlflow.log_param("hidden_layer_size", hidden_layer_size)
    mlflow.log_param("output_size", output_size)
    mlflow.log_param("dropout_prob", dropout_prob)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("patience", patience)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("seq_length", seq_length)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("outlier_handling", "Z-score")
    # mlflow.log_param("outlier_threshold", threshold)
    mlflow.log_param("scaling_method", "RobustScaler")

    # Log model parameters
    mlflow.pytorch.log_model(model, "model")

    # Train the model
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch, e_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch, e_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, e_batch in val_loader:
                y_pred = model(X_batch, e_batch)
                loss = loss_function(y_pred, y_batch)
                val_losses.append(loss.item())

        # Calculate average losses
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Log epoch metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate Model
    def evaluate(model, loader):
        model.eval()
        actuals, predictions = [], []
        with torch.no_grad():
            for X_batch, y_batch, e_batch in loader:
                y_pred = model(X_batch, e_batch)
                # Rescale y_batch values back to the original range
                actuals.extend(scaler.inverse_transform(y_batch.numpy().reshape(-1, 1)).flatten())
                # Rescale predictions back to the original range
                predictions.extend(scaler.inverse_transform(y_pred.numpy().reshape(-1, 1)).flatten())
        return np.array(actuals), np.array(predictions)

    train_actuals, train_predictions = evaluate(model, train_loader)
    val_actuals, val_predictions = evaluate(model, val_loader)

    print('Training set size:', len(train_actuals))
    print('Validation set size:', len(val_actuals))
    print('Training set predictions:', len(train_predictions))
    print('Validation set predictions:', len(val_predictions))

    # Metrics
    def calculate_metrics(actuals, predictions):
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        return mae, mse, rmse, mape

    train_mae, train_mse, train_rmse, train_mape = calculate_metrics(train_actuals, train_predictions)
    val_mae, val_mse, val_rmse, val_mape = calculate_metrics(val_actuals, val_predictions)

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

    plt.figure(figsize=(10, 5))
    plt.plot(train_actuals, label='Train Actuals')
    plt.plot(train_predictions, label='Train Predictions')
    plt.legend()
    plt.title('Train Set: Predictions vs Actuals')
    plt.savefig('train_predictions_vs_actuals.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_actuals, label='Validation Actuals')
    plt.plot(val_predictions, label='Validation Predictions')
    plt.legend()
    plt.title('Validation Set: Predictions vs Actuals')
    plt.savefig('val_predictions_vs_actuals.png')
    plt.close()

    # Log the plots as artifacts
    mlflow.log_artifact('train_predictions_vs_actuals.png')
    mlflow.log_artifact('val_predictions_vs_actuals.png')
    
    # Save model
    mlflow.pytorch.log_model(model, "best_model")

    # Define a custom Python model class for the scaler
    class ScalerWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, scaler):
            self.scaler = scaler

        def predict(self, context, model_input):
            return self.scaler.transform(model_input)

    # Wrap the scaler
    scaler_wrapper = ScalerWrapper(scaler)

    # Log the scaler as a custom Python model
    mlflow.pyfunc.log_model("scaler", python_model=scaler_wrapper)