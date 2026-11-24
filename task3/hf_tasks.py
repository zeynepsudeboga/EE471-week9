"""hf_tasks.py – one class that runs the ten Hugging Face tasks of Task 1.

Every method uses transformers.pipeline(), as in the Hugging Face LLM course.
The models are named explicitly (the same ones pipeline() picked by default in Task 1),
and each pipeline is loaded only the first time it is needed, then kept in memory.
"""
import gc

import torch
from transformers import pipeline


class HuggingFaceTasks:
    # task name -> (pipeline task, model on the Hugging Face Hub)
    MODELS = {
        "sentiment": ("sentiment-analysis", "distilbert/distilbert-base-uncased-finetuned-sst-2-english"),
        "zero_shot": ("zero-shot-classification", "facebook/bart-large-mnli"),
        "generation": ("text-generation", "HuggingFaceTB/SmolLM2-360M"),
        "fill_mask": ("fill-mask", "distilbert/distilroberta-base"),
        "ner": ("ner", "dbmdz/bert-large-cased-finetuned-conll03-english"),
        "qa": ("question-answering", "distilbert/distilbert-base-cased-distilled-squad"),
        "summarization": ("summarization", "sshleifer/distilbart-cnn-12-6"),
        "translation": ("translation", "Helsinki-NLP/opus-mt-fr-en"),
        "image": ("image-classification", "google/vit-base-patch16-224"),
        "speech": ("automatic-speech-recognition", "openai/whisper-large-v3"),
    }

    def __init__(self):
        self._pipelines = {}            # loaded pipelines, filled on first use

    def _get(self, name):
        """Return the pipeline for this task, loading it the first time (lazy loading).
        Only one model is kept in memory: all ten together do not fit in 8 GB of RAM."""
        if name not in self._pipelines:
            self._pipelines.clear()                  # forget the previous model to free its memory
            gc.collect()                             # delete it now
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()              # give the Mac GPU's memory back
            task, model = self.MODELS[name]
            print(f"Loading {model} ...")
            # torch_dtype="auto" keeps each model's own precision (Whisper is saved in float16)
            self._pipelines[name] = pipeline(task, model=model, torch_dtype="auto")
        return self._pipelines[name]

    # ---- text: understanding (encoder models) ----
    def sentiment(self, text):
        return self._get("sentiment")(text)[0]                       # {'label', 'score'}

    def zero_shot(self, text, labels):
        result = self._get("zero_shot")(text, candidate_labels=labels)
        return dict(zip(result["labels"], result["scores"]))

    def fill_mask(self, text, top_k=3):
        unmasker = self._get("fill_mask")
        text = text.replace("[MASK]", unmasker.tokenizer.mask_token)  # accept both mask styles
        return unmasker(text, top_k=top_k)

    def ner(self, text):
        return self._get("ner")(text, aggregation_strategy="simple")  # = grouped_entities=True

    def answer(self, question, context):
        return self._get("qa")(question=question, context=context)  # {'answer', 'score', ...}

    # ---- text: generation (decoder and encoder-decoder models) ----
    def generate(self, prompt, max_new_tokens=35, num_sequences=2):
        outputs = self._get("generation")(prompt, max_new_tokens=max_new_tokens,
                                          num_return_sequences=num_sequences, do_sample=True)
        return [o["generated_text"] for o in outputs]

    def summarize(self, text):
        return self._get("summarization")(text)[0]["summary_text"]

    def translate(self, text):
        return self._get("translation")(text)[0]["translation_text"]

    # ---- image and audio ----
    def classify_image(self, image):
        return self._get("image")(image)                              # path, URL or PIL image

    def transcribe(self, audio):
        return self._get("speech")(audio)["text"]                     # path or URL of an audio file


if __name__ == "__main__":
    tasks = HuggingFaceTasks()
    print(tasks.sentiment("I hate software courses"))
    print(tasks.translate("Ce cours est produit par Hugging Face."))
