import os
import re
import random

class SecondBrainAI:
    def __init__(self):
        """Initialize the Second Brain AI."""
        self.chat_history = []  # To maintain conversation context
        print("Custom Second Brain AI initialized!")

    def tokenize(self, text):
        """Simple tokenizer: splits text into lowercase words."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def generate_response(self, user_input):
        """Generate a response using custom logic."""
        # Tokenize input
        user_tokens = self.tokenize(user_input)

        # Basic keyword-based responses
        keywords = {
            "hello": "Hi there! How can I help you today?",
            "help": "Sure, let me know what you need assistance with.",
            "how": "I'm just a program, but I try my best to think like you!",
            "exit": "Goodbye! It was great chatting with you."
        }

        # Look for keywords in the user input
        for keyword, response in keywords.items():
            if keyword in user_tokens:
                return response

        # Default response
        return random.choice([
            "Tell me more about that.",
            "Why do you think so?",
            "Interesting! Let's dive deeper.",
        ])

    def clear_context(self):
        """Clear conversation history."""
        self.chat_history = []
        print("Context cleared. Starting fresh!")

    def respond(self, user_input):
        """Process user input, generate a response, and maintain context."""
        self.chat_history.append({"user": user_input})  # Add to history
        response = self.generate_response(user_input)
        self.chat_history.append({"second_brain": response})  # Add to history
        return response


if __name__ == "__main__":
    second_brain = SecondBrainAI()
    print("Welcome to your Second Brain AI! Type 'clear' to reset context or 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("Exiting... Have a great day!")
            break
        elif user_input.lower() == 'clear':
            second_brain.clear_context()
        else:
            response = second_brain.respond(user_input)
            print(f"Second Brain: {response}")
