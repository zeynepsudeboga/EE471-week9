from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
text = ("I am Nate, a research assistant at Izmir Institute of Technology, "
        "and currently living and working in the beautiful city İzmir in Türkiye.")
for e in ner(text):
    print(f"{e['entity_group']:4s} {e['word']:35s} {e['score']:.3f}")

    