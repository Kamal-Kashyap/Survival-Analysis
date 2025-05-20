import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from aft_model import build_aft_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Load processed data
data = pd.read_csv("raw_data.csv")

# Extract features and target
X = data.drop(columns=['left_interval', 'right_interval'])  # Feature columns
y = data[['left_interval', 'right_interval']]  # Interval-censored survival times

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
input_dim = X_train.shape[1]
model = build_aft_model(input_dim)

# Early stopping callback (prevents overfitting)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model and store training history
history = model.fit(
    X_train, y_train['left_interval'], 
    epochs=100, batch_size=32, 
    validation_data=(X_val, y_val['left_interval']), 
    callbacks=[early_stopping],  # 🔹 Added Early Stopping
    verbose=1
)

# Save the trained model
model.save("aft_model_trained.keras")  # 🔹 Use .keras instead of .h5
print("Training complete. Model saved as aft_model_trained.keras")

# ============================
# **✅ Step 2: Plot Loss Over Epochs**
# ============================
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss Over Epochs')
plt.show()

# ============================
# **✅ Step 3: Evaluate & Plot Actual vs Predicted**
# ============================

# Load the trained model
model = load_model("aft_model_trained.keras")

# Evaluate performance
loss, mae = model.evaluate(X_val, y_val['left_interval'], verbose=1)
print(f"Final Loss: {loss:.4f}, Final MAE: {mae:.4f}")

# Predict survival times
y_pred = model.predict(X_val)

# Scatter plot of Actual vs. Predicted survival time
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_val['left_interval'], y=y_pred.flatten())
plt.xlabel("Actual Survival Time")
plt.ylabel("Predicted Survival Time")
plt.title("Actual vs. Predicted Survival Time")
plt.show()
