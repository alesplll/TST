import streamlit as st
from models.t5_small import T5Rephraser


class PolitenessApp:
    def __init__(self):
        self.model = T5Rephraser()


def run(self):
    """Основной интерфейс приложения."""
    st.title("Text Politeness Transformer")
    st.write("Введите текст, чтобы преобразовать его в более вежливую форму.")

    input_text = st.text_area(
        "Введите текст", placeholder="Например: Это просто отвратительная работа!")
    if st.button("Преобразовать"):
        if input_text.strip():
            polite_text = self.model.transform_to_polite(input_text)
            st.subheader("Результат:")
            st.write(polite_text)
        else:
            st.warning("Пожалуйста, введите текст для преобразования.")
