import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model_name = "Ahmed-Ashraf-00/sentiment-model"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.config.id2label = {0: "Positive", 1: "Negative", 2: "Neutral", 3: "Irrelevant"}
    model.config.label2id = {"Positive": 0, "Negative": 1, "Neutral": 2, "Irrelevant": 3}
    return model, tokenizer

model, tokenizer = load_model()

st.title("🎭 Sentiment Analysis App")
st.write("Enter text and see whether it’s **Positive**, **Negative**, **Neutral**, or **Irrelevant**!")

text = st.text_area("✍️ Enter your text below:")

if st.button("Analyze"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item() * 100
        label = model.config.id2label[pred]
        st.subheader("Result:")
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.progress(confidence / 100)
    else:
        st.warning("Please enter some text to analyze!")
