from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def initialize_model(model_file="self_learning_model.h5", input_dim=2):
    """Initialize and save the machine learning model."""
    print("Initializing the model...")

    # Define the model architecture
    model = Sequential([
        Dense(32, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Output for binary classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Save the model
    model.save(model_file)
    print(f"Model saved to {model_file}!")


if __name__ == "__main__":
    initialize_model()