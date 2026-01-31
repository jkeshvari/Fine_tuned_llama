import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset

# 1. بارگذاری داده‌ها از فایل JSON
def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 2. پیش‌پردازش داده‌ها برای Fine-Tuning
def preprocess_data(data, tokenizer, max_length=512):
    processed_data = []
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # ترکیب دستورالعمل، ورودی و خروجی
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        response = output_text

        # Tokenization
        tokenized = tokenizer(prompt, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        labels = tokenizer(response, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]

        # Replace padding token labels with -100 to ignore in loss calculation
        labels[labels == tokenizer.pad_token_id] = -100

        processed_data.append({
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        })
    return processed_data

# 3. تعریف دیتاست سفارشی
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 4. بارگذاری مدل و توکنایزر
def load_model_and_tokenizer(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)  # استفاده از 8-bit برای کاهش مصرف حافظه
    return model, tokenizer

# 5. تنظیمات Fine-Tuning
def fine_tune(model, tokenizer, train_dataset, output_dir, epochs=3, batch_size=4):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        fp16=True,  # استفاده از محاسبات 16-bit برای بهینه‌سازی
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

# 6. اجرای Fine-Tuning
def main():
    # مسیر فایل داده و مدل
    json_file = "samp1.json"  # فایل داده JSON
    model_path = "C:\\Users\\dr.keshvari\\.cache\\huggingface\\hub\\models--bert-base-uncased"  # مسیر مدل LLaMA
    output_dir = "./fine_tuned_llama"  # مسیر خروجی مدل Fine-Tuned

    # بارگذاری داده‌ها
    raw_data = load_data(json_file)

    # بارگذاری مدل و توکنایزر
    model, tokenizer = load_model_and_tokenizer(model_path)

    # پیش‌پردازش داده‌ها
    processed_data = preprocess_data(raw_data, tokenizer)
    train_dataset = CustomDataset(processed_data)

    # اجرای Fine-Tuning
    fine_tune(model, tokenizer, train_dataset, output_dir)

if __name__ == "__main__":
    main()