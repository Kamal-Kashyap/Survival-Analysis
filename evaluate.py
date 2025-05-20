import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model_file="aft_model_trained.h5", data_file="processed_data.csv"):
    """Loads the trained AFT model and evaluates it on test data."""
    # Load processed data
    df = pd.read_csv(data_file)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    X_test = df[feature_cols]
    y_test = df['left_interval']  # Using left interval as the target for evaluation
    
    # Load trained model with explicit loss function
    model = tf.keras.models.load_model(model_file, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    
    print(f"Model Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

if __name__ == "__main__":
    evaluate_model()
