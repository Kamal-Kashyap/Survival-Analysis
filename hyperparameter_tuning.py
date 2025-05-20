import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd

def build_model(learning_rate=0.001, dropout_rate=0.2):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(10,)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def tune_hyperparameters(data_file="processed_data.csv"):
    """Performs hyperparameter tuning for the AFT model."""
    df = pd.read_csv(data_file)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    X = df[feature_cols]
    y = df['left_interval']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_mae = float("inf")
    best_params = {}
    
    for lr in [0.001, 0.0005, 0.0001]:
        for dropout in [0.2, 0.3, 0.4]:
            print(f"Training model with lr={lr}, dropout={dropout}")
            model = build_model(learning_rate=lr, dropout_rate=dropout)
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)
            val_mae = model.evaluate(X_val, y_val, verbose=0)[1]
            
            if val_mae < best_mae:
                best_mae = val_mae
                best_params = {"learning_rate": lr, "dropout_rate": dropout}
    
    print(f"Best Hyperparameters: {best_params} with MAE: {best_mae:.4f}")
    return best_params

if __name__ == "__main__":
    tune_hyperparameters()
