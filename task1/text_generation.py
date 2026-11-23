from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
print(generator(
    "If I continue to complete all in-class exercises in this modern software practices course,",
    max_new_tokens=35,          # max_length is overridden by the model's own generation_config
    num_return_sequences=2,
))
