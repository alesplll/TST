from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5SmallModel:
    def __init__(self, model_name="t5-small"):
        """Инициализация модели и токенайзера."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки модели: {e}")

    def transform_to_polite(self, text: str) -> str:
        """Преобразует текст в более вежливую форму."""
        if not text.strip():
            return "Ошибка: Входной текст пуст."

        try:
            # Формируем запрос с правильным prompt
            input_ids = self.tokenizer.encode(
                f"polite: {text}",
                return_tensors="pt",
                max_length=512,
                truncation=True
            )

            # Генерируем выход
            outputs = self.model.generate(
                input_ids,
                max_length=128,  # Устанавливаем разумное ограничение на длину
                num_beams=50,     # Используем Beam Search для лучшего качества
                early_stopping=True
            )

            # Возвращаем преобразованный текст
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            return f"Ошибка во время обработки текста: {e}"
