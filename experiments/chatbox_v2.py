#import streamlit as st
import pandas as pd
from transformers import pipeline, GPTJForCausalLM
import torch
from concurrent.futures import ThreadPoolExecutor
from model_loader import load_model, predict
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
from tqdm import tqdm


# example usage
sentiment_pipeline = pipeline('sentiment-analysis')

# text = "I'm lonely"
# result = sentiment_pipeline(text)

# print(result[0]['label'])

def create_classification_prompt(text):
    return f"Classify the mental health concern in the following message: {text}"

def analyze_sentiment(message):
    result = sentiment_pipeline(message)
    return result[0]['label']

def get_contextual_response(classification, sentiment):
    if classification == 1 and sentiment == 'NEGATIVE':
        return 'It sounds like you might be going through something serious. If you need support now, pls go directly to your nearest medical clinic or call this toll free emergency number 112 right now!'
    elif classification == 0 and sentiment == 'NEGATIVE':
        return "I'm sorry to hear that you're feeling this way. It's important to talk to someone who can provide support. Would you like me to find some resources for you?"
    elif sentiment == 'POSITIVE':
        return "I'm glad to hear that you're feeling positive! If there's anything else you need, feel free to let me know."
    else:
        return "Thank you for sharing. Is there anything specific you'd like to talk more?"

def process_in_batches(text_list, tokenizer, model, batch_size=16):
    all_features = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Processing Batches"):
        batch_texts = text_list[i:i + batch_size]
        tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**tokens)
            batch_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_features.append(batch_features)

    return np.concatenate(all_features, axis=0)

# features_balanced = process_in_batches(text.tolist(), tokenizer, model, batch_size=1)

def main():
    tokenizer, model = load_model()
    joblin_file = "logistic_regression_model.pkl"
    lmodel = joblib.load(joblin_file)
    text = ["Ouk yeah I feel like giving up on my plans for 2024 and also just leave this place Im living in "]
    # tokens = tokenizer(text)
    features_balanced = process_in_batches(text, tokenizer, model, batch_size=1)
    prediction = lmodel.predict(features_balanced)
    create_classification_prompt(text)
    label = analyze_sentiment(text)
    print(prediction)
    print(get_contextual_response(prediction[0], label))

if __name__ == "__main__":
    main()