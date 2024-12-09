#
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# model = AutoModelForCausalLM.from_config_file("meta-llama/Llama-3.2-1B")
# model.load_state_dict(torch.load("./fine_tuned_model_lora/lora_model_epoch_1.pt"))
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
# model.to(device)
# def generate_response(input_text):
#     model.eval()
#     inputs = tokenizer(input_text, return_tensors="pt").to(device)
#     outputs = model.generate(inputs["input_ids"], max_length=200, num_beams=5)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response
#
# example_input = "How do you know you are pregnant? Patient: I randomly get nauseatic and I have been constantly vomiting for the last few days. I also have a mild fever."
# print("Response:", generate_response(example_input))

from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer

# Load the pretrained encoder and decoder
encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-large-uncased", bos_token_id=101, eos_token_id=102)
decoder = BertGenerationDecoder.from_pretrained(
    "google-bert/bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
)

# Create the BERT2BERT model
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")

# Prepare the input text
input_text = "This is a long article to summarize."

# Tokenize the input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Perform inference
outputs = model.generate(
    input_ids=input_ids,
    max_length=50,          # Maximum output sequence length
    num_beams=5,            # Beam search for diversity
    early_stopping=True     # Stop when EOS token is reached
)

# Decode the output to text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input:", input_text)
print("Generated Summary:", generated_text)
