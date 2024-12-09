#%%
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict, Dataset
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import login
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import random
import numpy as np

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dataset = load_dataset("ruslanmv/ai-medical-chatbot")
train_dataset = dataset['train'].shuffle(seed=42)
num_partitions = 250
partition_size = len(train_dataset) // num_partitions
partitions = []
for i in range(num_partitions):
    start_idx = i * partition_size
    end_idx = start_idx + partition_size
    if i == num_partitions - 1:
        end_idx = len(train_dataset)
    partitions.append(train_dataset.select(range(start_idx, end_idx)))
print(set([len(partition) for partition in partitions]))

del dataset
del train_dataset

#%%
# Preprocess Dataset
def preprocess_data(example):
    input_text = example['Patient']
    output_text = example["Doctor"]
    return {"input_text": input_text, "output_text": output_text}
def tokenize_data(example):
    inputs = tokenizer(
        example["input_text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    outputs = tokenizer(
        example["output_text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs
def collate_fn(batch):
    input_ids = [torch.tensor(example["input_ids"]) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]
    labels = [torch.tensor(example["labels"]) for example in batch]

    # Pad sequences to make them of uniform length
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored tokens

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
def train_epoch(model, dataloader, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    # Create a tqdm progress bar for the training loop
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", ncols=100)

    for i, batch in progress_bar:
        # Move inputs and labels to the device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights after a set number of steps (gradient accumulation)
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        # Update the progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        # Create a tqdm progress bar for the evaluation loop
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating", ncols=100)

        for i, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss
def save_model(model, tokenizer, save_path="./fine_tuned_model_lora"):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
def save_model_as_pt(model, save_path="./fine_tuned_model/model.pt"):
    torch.save(model.state_dict(), save_path)
def load_model_from_pt(model_class, model_path, config_path):
    model = model_class.from_pretrained(config_path)
    model.load_state_dict(torch.load(model_path))
    return model


login("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") # Put your huggingface token here
# Tokenizer and Data Preparation
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=False , quantization_config=quantization_config,
                                             device_map="auto", offload_folder="offload"
                                             )
model.to(device)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
print("LoRA applied model:")
model.print_trainable_parameters()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

gradient_accumulation_steps = 8

for i , data in tqdm(enumerate(partitions)):
    processed_dataset = data.map(preprocess_data)
    tokenized_dataset = processed_dataset.map(tokenize_data, batched=True)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False , collate_fn=collate_fn)
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_epoch(model, train_loader, optimizer, device, gradient_accumulation_steps)
        evaluate_model(model, val_loader, device)
        save_model(model, tokenizer, f"./fine_tuned_model_lora_epoch_{epoch + 1}")
    model.save_pretrained(f"./fine_tuned_model_lora{i}_partition")
    tokenizer.save_pretrained(f"./fine_tuned_model_lora{i}_partition")
    save_model_as_pt(model, f"./fine_tuned_model_lora/lora_model_epoch_{i}_partition.pt")

# torch.save(model, f"./fine_tuned_model_lora/lora_model.pt")
