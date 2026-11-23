from transformers import pipeline

unmasker = pipeline("fill-mask")
print(unmasker("To understand generative AI, one must study <mask> well.", top_k=2))
