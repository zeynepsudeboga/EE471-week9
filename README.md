# EE471 – Week 9: Hugging Face Transformers

Zeynep Sude Boğa – 290206047

- `task1/` – the ten pipeline() tasks (one script each)
- `task3/` – `hf_tasks.py` (OOP module) and `app.py` (Gradio demo): `python app.py`
- `task4/` – tokenizer basics: load/save, encode/decode, padding, truncation, special tokens

Setup: `pip install -r requirements.txt`. transformers 4 is used because transformers 5 removed the question-answering pipeline; Gradio 5 because Gradio 6 needs huggingface-hub 1.x, which transformers 4 does not support.
