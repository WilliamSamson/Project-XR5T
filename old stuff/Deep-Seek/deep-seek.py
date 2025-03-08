import argparse
import torch

# Check if a GPU is available. DeepSeek-R1 FP8 quantization requires a GPU.
if not torch.cuda.is_available():
    print("Error: No GPU found. DeepSeek-R1 requires a GPU for FP8 quantization. Please run on a GPU-enabled machine.")
    exit(1)

from transformers import pipeline


class DeepSeekChatInterface:
    """
    Chat interface using the Transformers pipeline with DeepSeek-R1.
    It instructs the model to "think step-by-step" (chain-of-thought) before providing the final answer.
    """

    def __init__(self, model_name="deepseek-ai/DeepSeek-R1", enable_cot=True):
        self.enable_cot = enable_cot
        print("Loading text-generation pipeline...")
        self.chat_pipeline = pipeline(
            "text-generation",
            model=model_name,
            trust_remote_code=True,
            device=0  # Use the first GPU
        )
        print("Pipeline loaded successfully.")

    def generate_response(self, prompt, max_length=256):
        """
        Generate a response for the given prompt.
        When chain-of-thought is enabled, the model is instructed to reveal its reasoning.

        Args:
            prompt (str): The user's prompt.
            max_length (int): Maximum length for the generated response.

        Returns:
            tuple: (chain_of_thought, final_answer)
        """
        if self.enable_cot:
            full_prompt = (
                f"You are an advanced AI assistant. Think step-by-step before answering.\n"
                f"User: {prompt}\n"
                f"[Chain-of-Thought]:"
            )
        else:
            full_prompt = f"User: {prompt}\nAI:"

        # Generate text using the pipeline
        results = self.chat_pipeline(
            full_prompt,
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7
        )
        generated_text = results[0]['generated_text']

        if self.enable_cot:
            # Expecting the model to separate reasoning and final answer with "[Final Answer]:"
            if "[Final Answer]:" in generated_text:
                parts = generated_text.split("[Final Answer]:", 1)
                chain_of_thought = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                chain_of_thought = "No explicit chain-of-thought available."
                final_answer = generated_text.strip()
        else:
            chain_of_thought = ""
            final_answer = generated_text.strip()

        return chain_of_thought, final_answer

    def chat_loop(self):
        """
        Runs an interactive session that accepts user prompts and outputs both the chain-of-thought and final answer.
        """
        print("\n=== DeepSeek R1 Chat Interface using Pipeline ===")
        print("Type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat. Goodbye!")
                break

            if user_input.lower() == "exit":
                print("Exiting chat. Goodbye!")
                break

            cot, answer = self.generate_response(user_input)
            if self.enable_cot:
                print("\n[Chain-of-Thought]:")
                print(cot)
            print("\n[Final Answer]:")
            print(answer)
            print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepSeek R1 Chat Interface using Pipeline with Chain-of-Thought feature"
    )
    parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought feature")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1", help="Model identifier for DeepSeek-R1")
    args = parser.parse_args()

    chat_interface = DeepSeekChatInterface(model_name=args.model, enable_cot=not args.no_cot)
    chat_interface.chat_loop()
