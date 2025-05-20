import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_aft_model(input_dim, dropout_rate=0.3):  # Add dropout_rate parameter
    """Builds a neural network-based AFT model with dropout."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1(0.01)),  
        layers.Dropout(dropout_rate),  # Dropout added here
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout_rate),  # Dropout added again
        layers.Dense(1, activation='linear')  
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse',
                  metrics=['mae'])
    
    return model

if __name__ == "__main__":
    # Example usage
    input_dim = 2  # Number of features
    model = build_aft_model(input_dim)
    model.summary()
