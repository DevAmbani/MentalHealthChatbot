import torch
from transformers import BertTokenizer, BertModel

def load_model():
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Path to your saved model weights
    # model_path_ = model_path

    # # Load the model weights
    # model.load_state_dict(torch.load(model_path))
    return tokenizer, model

def predict(text, tokenizer, model):
    model.eval()
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
        features = outputs.last_hidden_state.mean(dim=1)
    return features

# example usage
# tokenizer, model = load_model()
# text = "Example text for prediction."
# prediction = predict(text, tokenizer, model)
# print(prediction)