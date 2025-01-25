import os
import json
import random
import numpy as np
import requests
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import schedule
import time

class SelfLearningAI:
    def __init__(self, model_file="self_learning_model.h5", data_file="data.json"):
        self.model_file = model_file
        self.data_file = data_file
        self.model = self._initialize_or_load_model()
        self.data = self._load_or_initialize_data()
        self.input_dim = 2  # Default input dimensions

    def _initialize_or_load_model(self):
        """Load an existing model or initialize a new one."""
        if os.path.exists(self.model_file):
            print("Loading existing model...")
            return load_model(self.model_file)
        print("Initializing a new model...")
        model = Sequential([
            Dense(32, input_dim=self.input_dim, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # Output for binary classification
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _load_or_initialize_data(self):
        """Load existing data or create an empty dataset."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as file:
                return json.load(file)
        return {"inputs": [], "outputs": []}

    def _save_model(self):
        """Save the current state of the model."""
        self.model.save(self.model_file)

    def _save_data(self):
        """Save the training data to a file."""
        with open(self.data_file, 'w') as file:
            json.dump(self.data, file)

    def _add_data_point(self, inputs, output):
        """Add a new data point to the dataset."""
        self.data["inputs"].append(inputs)
        self.data["outputs"].append(output)

    def _generate_synthetic_data(self, num_samples=50):
        """Generate synthetic training data based on predefined rules."""
        print("Generating synthetic data...")
        for _ in range(num_samples):
            x1 = random.uniform(0, 10)
            x2 = random.uniform(0, 10)
            output = 1 if x1 + x2 > 10 else 0  # Simple rule for classification
            self._add_data_point([x1, x2], output)

    def _fetch_real_world_data(self):
        """Fetch real-world data from an API (e.g., weather data)."""
        print("Fetching real-world data...")
        api_url = "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current_weather=true"
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                temp = data["current_weather"]["temperature"]
                wind_speed = data["current_weather"]["windspeed"]
                output = 1 if temp > 15 else 0  # Example rule
                self._add_data_point([temp, wind_speed], output)
                print(f"Fetched: Temp={temp}, Wind Speed={wind_speed}, Output={output}")
            else:
                print("Failed to fetch data from API.")
        except Exception as e:
            print(f"Error fetching data: {e}")

    def retrain_model(self, epochs=10):
        """Retrain the model using the current dataset."""
        if len(self.data["inputs"]) < 10:
            print("Not enough data to retrain. Add more examples.")
            return
        inputs = np.array(self.data["inputs"])
        outputs = np.array(self.data["outputs"])
        print("Retraining the model...")
        self.model.fit(inputs, outputs, epochs=epochs, verbose=1)
        self._save_model()

    def predict(self, input_data):
        """Make predictions using the trained model."""
        input_data = np.array(input_data).reshape(1, -1)
        prediction = self.model.predict(input_data, verbose=0)
        return prediction[0][0]

    def interactive_session(self):
        """Run an interactive session for user input and AI predictions."""
        print("Welcome to Self-Learning AI! Type 'exit' to quit.")
        while True:
            user_input = input("Enter input data (comma-separated) or type 'fetch'/'generate': ").strip()
            if user_input.lower() == 'exit':
                print("Saving knowledge and exiting...")
                self._save_data()
                break
            elif user_input.lower() == 'fetch':
                self._fetch_real_world_data()
                self.retrain_model()
            elif user_input.lower() == 'generate':
                self._generate_synthetic_data()
                self.retrain_model()
            else:
                try:
                    input_data = list(map(float, user_input.split(',')))
                    prediction = self.predict(input_data)
                    print(f"AI Prediction: {prediction:.2f}")
                except ValueError:
                    print("Invalid input. Please provide numeric data.")

    def schedule_tasks(self):
        """Schedule periodic tasks for continuous learning and retraining."""
        schedule.every(5).minutes.do(self._generate_synthetic_data)
        schedule.every(10).minutes.do(self._fetch_real_world_data)
        schedule.every(15).minutes.do(self.retrain_model)
        print("Scheduled tasks are running...")
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    ai = SelfLearningAI()
    ai.interactive_session()
