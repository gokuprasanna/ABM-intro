import mesa
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMAgent(mesa.Agent):
    def __init__(self, unique_id, model, model_name="gpt2-medium"):
        super().__init__(unique_id, model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.conversation_history = []

    def generate_response(self, input_text, max_length=100):
        self.conversation_history.append(f"User: {input_text}")
        full_input = " ".join(self.conversation_history)
        input_ids = self.tokenizer.encode(full_input, return_tensors="pt")
        
        with torch.no_grad():
            output = self.llm.generate(input_ids, max_length=max_length, num_return_sequences=1)
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        new_response = response[len(full_input):].strip()
        self.conversation_history.append(f"Agent: {new_response}")
        return new_response

    def step(self):
        # Define the agent's behavior during each step of the simulation
        pass
