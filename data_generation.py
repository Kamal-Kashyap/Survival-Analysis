import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000, n_features=2):
    """Generates synthetic interval-censored survival data."""
    np.random.seed(42)
    
    # Generate feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Generate true survival times
    true_times = np.exp(0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n_samples))
    
    # Generate interval-censored observations
    left_censoring = np.random.uniform(0.7, 1.0, size=n_samples)
    right_censoring = np.random.uniform(1.0, 1.3, size=n_samples)
    
    left_interval = true_times * left_censoring
    right_interval = true_times * right_censoring
    
    # Create a DataFrame
    columns = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['left_interval'] = left_interval
    df['right_interval'] = right_interval
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("raw_data.csv", index=False)
    print("Synthetic data saved as raw_data.csv")
