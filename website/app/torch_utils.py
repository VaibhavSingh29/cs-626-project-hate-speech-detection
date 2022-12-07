import torch
from PIL import Image
from trainer import Preprocessor, Classifier

MBERT_CHECKPOINT = 'bert-base-multilingual-cased'
MBERT_TOKEN_LENGTH = 128

# load model
mbert = Classifier(MBERT_CHECKPOINT)
mbert.load_state_dict(torch.load(
    '../../saved_models/bert-base-multilingual-cased-60.93.pt'))

# preprocess sentence
preprocessor = Preprocessor(MBERT_CHECKPOINT, MBERT_TOKEN_LENGTH)


def get_encoding(sentence):
    encoded_sent = preprocessor.process_one(sentence)
    return encoded_sent['input_ids'].reshape(1, MBERT_TOKEN_LENGTH), encoded_sent['attention_mask'].reshape(1, MBERT_TOKEN_LENGTH)


def get_prediction(input_ids, attention_mask):
    prob = mbert.predict_prob(input_ids, attention_mask)
    return prob
