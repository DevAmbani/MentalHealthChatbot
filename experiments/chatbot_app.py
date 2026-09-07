import streamlit as st
import pandas as pd
from transformers import GPTJForCausalLM, AutoTokenizer, pipeline
import torch
from concurrent.futures import ThreadPoolExecutor
import emoji

# Load preprocessed data
data = pd.read_csv('preprocessed_data_cleaned.csv', dtype=str)

# Initialize the GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize sentiment analysis model
sentiment_pipeline = pipeline('sentiment-analysis')

# Define functions to interact with GPT-J
def create_classification_prompt(text):
    return f"Classify the mental health concern in the following message: {text}"

def create_sentiment_prompt(text):
    return f"Analyze the sentiment of the following message: {text}"

def gptj_completion(prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def classify_message(message):
    prompt = create_classification_prompt(message)
    classification = gptj_completion(prompt)
    return classification

def analyze_sentiment(message):
    result = sentiment_pipeline(message)
    return result[0]['label'], result[0]['score']

def get_contextual_response(classification, sentiment, message):
    if 'acute' in classification.lower():
        return "It sounds like you might be going through something serious. Please reach out to a trusted person or contact a helpline immediately."
    elif sentiment == 'NEGATIVE' and 'depression' in classification.lower():
        return "I'm sorry to hear that you're feeling this way. It's important to talk to someone who can provide support. Would you like me to find some resources for you?"
    elif sentiment == 'POSITIVE':
        return "I'm glad to hear that you're feeling positive! If there's anything else you need, feel free to let me know."
    else:
        return "Thank you for sharing. Is there anything specific you'd like to talk about?"

def process_message(message):
    classification = classify_message(message)
    sentiment, score = analyze_sentiment(message)
    response = get_contextual_response(classification, sentiment, message)
    return classification, sentiment, score, response

# Streamlit app layout
st.title("Mental Health Chatbot")
st.write("This chatbot helps to classify mental health concerns and analyze sentiment in messages.")

# Chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# User input
user_input = st.text_area("Enter your message:", "", key="input_area")

if st.button("Send"):
    if user_input:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_message, user_input)
            classification, sentiment, score, response = future.result()

        # Append user query and bot response to history
        st.session_state['history'].append({"user": user_input, "bot": response})

        # Clear input
        st.session_state['input_area'] = ""

# Display chat history
for chat in st.session_state['history']:
    st.chat_message("user").markdown(chat['user'])
    st.chat_message("bot").markdown(chat['bot'])
