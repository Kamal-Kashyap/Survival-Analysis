import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd

def build_model(input_dim, learning_rate=0.0001, dropout_rate=0.3):  # Updated hyperparameters
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def train_final_model(data_file="processed_data.csv"):
    df = pd.read_csv(data_file)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    X = df[feature_cols]
    y = df['left_interval']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(input_dim=X_train.shape[1])  # Uses best hyperparameters
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    val_mae = model.evaluate(X_val, y_val, verbose=1)[1]
    print(f"Final Model MAE: {val_mae:.4f}")
    model.save("final_aft_model.h5")

if __name__ == "__main__":
    train_final_model()
