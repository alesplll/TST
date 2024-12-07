from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5Rephraser:
    def __init__(self, model_name="flax-community/t5-rephrasing"):
        """Инициализация модели и токенайзера."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def rephrase(self, text: str) -> str:
        """Перефразирует текст, делая его более вежливым."""
        # Формируем запрос для перефразирования
        input_ids = self.tokenizer.encode(
            f"paraphrase: {text}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        # Генерируем выходные данные с помощью модели
        outputs = self.model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        # Декодируем результат в текст
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
