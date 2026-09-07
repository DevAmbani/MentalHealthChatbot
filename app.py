import chainlit as cl
from transformers import pipeline, BertTokenizer, BertModel
import torch
import joblib
import numpy as np
from tqdm import tqdm

# Pinned so a HF default change can't silently alter behavior.
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_REVISION = "714eb0f"
MODEL_PATH = "logistic_regression_model.pkl"

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    revision=SENTIMENT_REVISION,
)


def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model


def analyze_sentiment(message):
    result = sentiment_pipeline(message)
    return result[0]["label"]


def get_contextual_response(classification, sentiment):
    if classification == 1 and sentiment == "NEGATIVE":
        return (
            "It sounds like you might be going through something serious. "
            "If you need support now, please go directly to your nearest medical "
            "clinic or call this toll-free emergency number 112 right now!"
        )
    elif classification == 0 and sentiment == "NEGATIVE":
        return (
            "I'm sorry to hear that you're feeling this way. It's important to "
            "talk to someone who can provide support. Would you like me to find "
            "some resources for you?"
        )
    elif sentiment == "POSITIVE":
        return (
            "I'm glad to hear that you're feeling positive! If there's anything "
            "else you need, feel free to let me know."
        )
    else:
        return "Thank you for sharing. Is there anything specific you'd like to talk more about?"


def process_in_batches(text_list, tokenizer, model, batch_size=16):
    all_features = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Processing Batches"):
        batch_texts = text_list[i : i + batch_size]
        tokens = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**tokens)
            batch_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_features.append(batch_features)

    return np.concatenate(all_features, axis=0)


lmodel = joblib.load(MODEL_PATH)
tokenizer, model = load_model()


@cl.on_message
async def handle_message(message):
    text = [message.content]
    features = process_in_batches(text, tokenizer, model, batch_size=1)
    prediction = lmodel.predict(features)
    sentiment_label = analyze_sentiment(message.content)
    response = get_contextual_response(prediction[0], sentiment_label)
    await cl.Message(content=response).send()


if __name__ == "__main__":
    cl.run()
