from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5SmallModel:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def transform_to_polite(self, text: str) -> str:
        """Преобразует текст в более вежливую форму."""
        input_ids = self.tokenizer.encode(
            f"make sentences more polite: {text}", return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            input_ids, max_length=512, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
