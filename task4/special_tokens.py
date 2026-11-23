from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentence = "I've been waiting for a HuggingFace course my whole life."

ids_without = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
ids_with = tokenizer(sentence)["input_ids"]                  # calling the tokenizer adds them
print("Without:", ids_without)
print("With   :", ids_with)
print(tokenizer.decode(ids_with))
