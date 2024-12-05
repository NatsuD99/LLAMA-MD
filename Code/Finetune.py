from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
# 1. Load the Dataset
dataset = load_dataset("ruslanmv/ai-medical-chatbot")

# Preprocess the dataset
def preprocess_data(example):
    return {
        "input_text": f"{example['Description']} Patient: {example['Patient']}",
        "output_text": example["Doctor"]
    }

processed_dataset = dataset.map(preprocess_data)
train_test_split = processed_dataset["train"].train_test_split(test_size=0.1)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples["output_text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# 4. Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer
)

trainer.train()

# 5. Save and Evaluate the Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")


text = "Describe the symptoms of flu. Patient: I have fever and cough."
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model.generate(inputs["input_ids"], max_length=200, num_beams=5)
print("Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))