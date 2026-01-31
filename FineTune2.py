from llama_factory import LlamaFactory
import json
from torch.utils.data import Dataset, DataLoader

# 1. بارگذاری داده‌ها از فایل JSON
def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 2. دیتاست سفارشی برای داده‌ها
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # ساختن prompt
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        response = output_text

        # Tokenization
        inputs = self.tokenizer(prompt, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        labels = self.tokenizer(response, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]

        # جایگزینی توکن‌های padding با -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

# 3. اجرای Fine-Tuning
def fine_tune_llama(model_path, json_file, output_dir, batch_size=4, epochs=3, max_length=512):
    # بارگذاری داده‌ها
    data = load_data(json_file)

    # بارگذاری مدل و توکنایزر با استفاده از LlamaFactory
    llama = LlamaFactory(model_path)
    tokenizer = llama.tokenizer
    model = llama.model

    # ایجاد دیتاست و DataLoader
    dataset = CustomDataset(data, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # تنظیمات Fine-Tuning
    llama.fine_tune(
        dataloader=dataloader,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=2e-5
    )

# 4. اجرای برنامه
if __name__ == "__main__":
    # مسیر فایل داده و مدل
    json_file = "data.json"  # فایل داده JSON
    model_path = "path_to_local_llama_model"  # مسیر مدل LLaMA
    output_dir = "./fine_tuned_llama"  # مسیر خروجی مدل Fine-Tuned

    # اجرای Fine-Tuning
    fine_tune_llama(model_path, json_file, output_dir)