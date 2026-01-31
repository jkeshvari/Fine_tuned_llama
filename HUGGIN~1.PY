from transformers import AutoModel, AutoTokenizer

# دانلود مدل
model = AutoModel.from_pretrained("bert-base-uncased")

# دانلود توکنایزر
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")