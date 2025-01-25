import json
import random

def generate_data(file_name="data.json", num_samples=100):
    """Generate initial synthetic data for training."""
    data = {"inputs": [], "outputs": []}

    print("Generating synthetic data...")
    for _ in range(num_samples):
        x1 = random.uniform(0, 10)  # Random value between 0 and 10
        x2 = random.uniform(0, 10)
        output = 1 if x1 + x2 > 10 else 0  # Classification rule
        data["inputs"].append([x1, x2])
        data["outputs"].append(output)

    # Save the data to a JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file)
    print(f"Data saved to {file_name}!")

if __name__ == "__main__":
    generate_data()