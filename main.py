import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from tqdm import tqdm
import torch.nn.functional as F
import os
from torch.optim import AdamW

# CONFIG 
# roberta-large for better performance
MODEL_NAME = "roberta-large"
MAX_LEN = 256
BATCH_SIZE = 4 
EPOCHS = 10
LR = 1e-5 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Centralized file paths (make sure to change accordingly)
TRAIN_FILE_PATH = "/content/ethos_ml/train.csv"
TEST_FILE_PATH = "/content/ethos_ml/test.csv"
OUTPUT_DIR = "/content/ethos_ml/"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "roberta_finetuned_model")

print(f"\n🚀 Using device: {DEVICE.upper()}")

# DATASET
class QA_Dataset(Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use a separator token for Roberta for better performance
        text = f"Topic: {row['topic']}{tokenizer.sep_token}Problem: {row['problem_statement']}"
        options = [row[f"answer_option_{i}"] for i in range(1, 6)]
        
        # Correctly format for roberta: [[text, opt1], [text, opt2], ...]
        text_options_pairs = [[text, str(opt)] for opt in options]
        
        inputs = self.tokenizer(
            text_options_pairs,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        if self.is_train:
            label = int(row["correct_option_number"]) - 1
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": torch.tensor(label),
            }
        else:
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "topic": row["topic"],
                "problem_statement": row["problem_statement"],
                "options": options
            }

# LOAD DATA 
print("\n📂 Loading data...")
train_df = pd.read_csv("/content/ethos_ml/train.csv")
test_df = pd.read_csv("/content/ethos_ml/test.csv")

# Using RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

train_dataset = QA_Dataset(train_df, tokenizer, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL SETUP 
# Using RobertaForMultipleChoice
print(f"\nSetting up model: {MODEL_NAME}...")
model = RobertaForMultipleChoice.from_pretrained(MODEL_NAME)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# TRAINING 
print("\n🎯 Fine-tuning RoBERTa on training data...\n")
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        # Move batch to device
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

print("\n✅ Training complete!")

#  SAVE THE FINETUNED MODEL 
print(f"\n💾 Saving finetuned model to {MODEL_SAVE_PATH}...")
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("✅ Model saved successfully!")


# TEST PREDICTION 
print("\n🧠 Generating predictions on test data...")
model.eval()

test_dataset = QA_Dataset(test_df, tokenizer, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # batch_size=1 for prediction

predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE) 
        attn_mask = batch["attention_mask"].to(DEVICE) 
        topic = batch["topic"][0]
        problem = batch["problem_statement"][0]
        options = [opt[0] for opt in batch["options"]]

        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        
        pred_option = torch.argmax(outputs.logits).item() + 1

        predictions.append({
            "topic": topic,
            "problem_statement": problem,
            "solution": options[pred_option - 1],
            "correct_option": pred_option
        })
        
#  SAVE OUTPUT 
output_df = pd.DataFrame(predictions)
output_path = "./output.csv" #change path accordingly
output_df.to_csv(output_path, index=False)

print(f"\n📁 Output saved to {output_path}")
print("Sample output:")
print(output_df.head())