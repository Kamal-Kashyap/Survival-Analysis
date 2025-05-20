import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader  # Added this import
from survival_model import IntervalCensoredDataset, DeepSurvivalNet, train_model

def analyze_model_outputs(model, test_loader):
    print("Starting analysis...")  # Added print statement
    model.eval()
    all_hazards = []
    all_lower_times = []
    all_upper_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            hazard_pred = model(features)
            all_hazards.extend(hazard_pred.numpy())
            all_lower_times.extend(batch['lower_time'].numpy())
            all_upper_times.extend(batch['upper_time'].numpy())
    
    hazards = np.array(all_hazards)
    lower_times = np.array(all_lower_times)
    upper_times = np.array(all_upper_times)
    
    print("Generating survival curves...")  # Added print statement
    time_points = np.linspace(0, max(upper_times), 100)
    survival_curves = np.exp(-np.outer(hazards, time_points))
    
    print("Creating plot...")  # Added print statement
    plt.figure(figsize=(12, 8))
    
    for i in range(min(10, len(survival_curves))):
        plt.plot(time_points, survival_curves[i], alpha=0.3, color='blue')
    
    plt.plot(time_points, survival_curves.mean(axis=0), 
             color='red', linewidth=2, label='Mean Survival')
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Predicted Survival Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return hazards, survival_curves

def main():
    print("Starting main function...")  # Added print statement
    # Generate sample data
    n_samples = 1000
    n_features = 10
    
    print("Generating sample data...")  # Added print statement
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    base_times = np.random.exponential(50, n_samples)
    lower_times = base_times * np.random.uniform(0.8, 0.9, n_samples)
    upper_times = base_times * np.random.uniform(1.1, 1.2, n_samples)
    
    print("Creating dataset...")  # Added print statement
    dataset = IntervalCensoredDataset(features, lower_times, upper_times)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print("Initializing model...")  # Added print statement
    model = DeepSurvivalNet(input_dim=n_features)
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, epochs=100)
    
    print("\nAnalyzing model predictions...")
    hazards, survival_curves = analyze_model_outputs(model, test_loader)
    
    print("\nPrediction Summary:")
    print(f"Mean predicted hazard rate: {hazards.mean():.4f}")
    print(f"Median predicted hazard rate: {np.median(hazards):.4f}")
    print(f"Hazard rate std deviation: {hazards.std():.4f}")
    
    median_survival = np.median(survival_curves, axis=0)
    median_survival_time = np.interp(0.5, median_survival[::-1], 
                                   np.linspace(0, max(upper_times), 100)[::-1])
    print(f"Predicted median survival time: {median_survival_time:.2f}")
    print("\nDone!")  # Added print statement

if __name__ == "__main__":
    main()