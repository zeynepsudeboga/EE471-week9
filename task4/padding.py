from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

batch = tokenizer(sequences, padding=True)          # pad the shorter sentence to the longest one
for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
    print("input_ids     :", ids)
    print("attention_mask:", mask, "\n")
