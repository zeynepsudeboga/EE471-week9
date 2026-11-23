from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentence = "I've been waiting for a HuggingFace course my whole life."

full = tokenizer(sentence)
cut = tokenizer(sentence, max_length=8, truncation=True)     # keep at most 8 tokens
print("Full     :", len(full["input_ids"]), "tokens ->", tokenizer.decode(full["input_ids"]))
print("Truncated:", len(cut["input_ids"]), "tokens ->", tokenizer.decode(cut["input_ids"]))
