import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.2-1B"
# model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)


def doctor_response(question, max_length=200, temperature=0.5):
    """
    Generates a doctor's response based on the patient's question.

    Parameters:
        question (str): The patient's question.
        max_length (int): The maximum token length of the response.
        temperature (float): Sampling temperature for diversity.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The doctor's response.
    """
    prompt = (
              "<System Prompt>"
              "You are a Doctor specializing in Gynaecology. Please diagnose this Patient. Answer specifically to the patient's question."
              "</SystemPrompt>"
              "Patient: {question}\n"
              "Doctor:" .format(question=question)
              )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response # Extract the response after "Doctor:"


# Example usage
if __name__ == "__main__":
    print("Welcome to the virtual doctor's assistant!")
    # user_input = input("You (Patient): ")
    user_input = "I randomly get nauseatic and I have been constantly vomiting for the last few days. I also have a mild fever. What do you think is wrong with me?"
    answer = doctor_response(user_input)
    print(answer)
