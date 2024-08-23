import os
import numpy as np
import pickle
import logging
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Sequential
from gensim.models.ldamodel import LdaModel
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
from transformers import BertTokenizer
from model_definition import BERTClassifier

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load necessary models and components
logging.info("Loading models and components...")
with open('../models/dad_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
embedding_matrix = np.load('../models/dad_embedding_matrix.npy')
with open('../models/bow_vectorizer.pkl', 'rb') as handle:
    bow_vectorizer = pickle.load(handle)
with open('../models/tfidf_vectorizer.pkl', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)
lda_model = LdaModel.load('../models/lda_model.gensim')
dictionary = Dictionary.load('../models/lda_model.gensim.id2word')
word2vec_model = Word2Vec.load("../models/dad_word2vec.model")

# Load BERT tokenizer and models
bert_tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
best_models_bert = {}
for emotion in ["شعور الفرح - Joy", "شعور الحزن – Sadness ", "شعور الخوف – Fear", "شعور الحب - Liking", "شعور البغض - Hatred"]:
    model_path = f'../models/best_model_bert_{emotion}.pt'
    if os.path.exists(model_path):
        model = BERTClassifier()  # Create an instance of your custom class
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode
        best_models_bert[emotion] = model
    else:
        print(f"Model file not found for {emotion}: {model_path}")

maxlen = 100
embedding_dim = 300
expected_num_features_lda = 100  # Number of topics

# Load best models for each approach
logging.info("Loading best models for each emotion...")
best_models = {}
emotions = [
    "شعور البغض - Hatred",
    "شعور الحب - Liking",
    "شعور الحزن – Sadness ", 
    "شعور الخوف – Fear",
    "شعور الفرح - Joy"
]

for emotion in emotions:
    try:
        best_models[emotion] = {
            "lda": load_model(f'../models/best_model_lda_{emotion}.h5'),
            "word2vec": load_model(f'../models/best_model_word2vec_{emotion}.h5'),
            "tfidf": load_model(f'../models/best_model_tfidf_{emotion}.h5'),
            "bow": load_model(f'../models/best_model_bow_{emotion}.h5'),
            "tf_tokenizer": load_model(f'../models/best_model_{emotion}.h5')
        }
    except:
        best_models[emotion] = {
            "lda": pickle.load(open(f'../models/best_model_lda_{emotion}.pkl', 'rb')),
            "word2vec": pickle.load(open(f'../models/best_model_word2vec_{emotion}.pkl', 'rb')),
            "tfidf": pickle.load(open(f'../models/best_model_tfidf_{emotion}.pkl', 'rb')),
            "bow": pickle.load(open(f'../models/best_model_bow_{emotion}.pkl', 'rb')),
            "tf_tokenizer": pickle.load(open(f'../models/best_model_{emotion}.pkl', 'rb'))
        }

# Preprocess functions
def preprocess_text(text, maxlen):
    logging.debug(f"Preprocessing text for tokenization: {text}")
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

def preprocess_text_lda(text, num_topics):
    logging.debug(f"Preprocessing text for LDA: {text}")
    tokens = text.split()
    bow = dictionary.doc2bow(tokens)
    lda_features = [0] * num_topics
    for idx, prob in lda_model.get_document_topics(bow, minimum_probability=0):
        lda_features[idx] = prob
    if len(lda_features) < num_topics:
        lda_features.extend([0] * (num_topics - len(lda_features)))
    return np.array([lda_features])

def preprocess_text_tfidf(text):
    logging.debug(f"Preprocessing text for TF-IDF: {text}")
    return tfidf_vectorizer.transform([text]).toarray()

def preprocess_text_bow(text):
    logging.debug(f"Preprocessing text for BoW: {text}")
    return bow_vectorizer.transform([text]).toarray()

def preprocess_text_word2vec(text, model, embedding_dim):
    logging.debug(f"Preprocessing text for Word2Vec: {text}")
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.array([np.mean(word_vectors, axis=0)])
    else:
        return np.zeros((1, embedding_dim))

def load_test_data(csv_path):
    df = pd.read_csv(csv_path)
    X_test = []
    y_test = []

    for column in df.columns[1:]:  # Skip the first column assuming it's an index or ID
        texts = df[column].dropna().tolist()
        X_test.extend(texts)
        y_test.extend([column] * len(texts))
    
    return X_test, y_test

def preprocess_text_bert(text, tokenizer, max_length):
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True
    )
    input_ids = np.array([inputs['input_ids']])
    attention_mask = np.array([inputs['attention_mask']])
    return input_ids, attention_mask

# Prediction function
def predict_emotion(text):
    logging.info(f"Predicting emotion for text: {text}")
    
    # Preprocess the text for each model type
    processed_text_blank = preprocess_text(text, maxlen)
    processed_text_lda = preprocess_text_lda(text, expected_num_features_lda)
    processed_text_tfidf = preprocess_text_tfidf(text)
    processed_text_bow = preprocess_text_bow(text)
    processed_text_word2vec = preprocess_text_word2vec(text, word2vec_model, embedding_dim)
    input_ids_bert, attention_mask_bert = preprocess_text_bert(text, bert_tokenizer, maxlen)

    # Initialize results
    emotion_scores = {emotion: [] for emotion in emotions}
    emotion_votes = {emotion: 0 for emotion in emotions}

    # Predict using each model type
    for emotion in emotions:
        logging.debug(f"Predicting for emotion: {emotion}")
        for model_type, model in best_models[emotion].items():
            logging.debug(f"Using model type: {model_type}")
            if model_type == "lda":
                prediction = model.predict(processed_text_lda)
            elif model_type == "word2vec":
                prediction = model.predict(processed_text_word2vec)
            elif model_type == "tfidf":
                prediction = model.predict(processed_text_tfidf)
            elif model_type == "bow":
                prediction = model.predict(processed_text_bow)
            else:
                prediction = model.predict(processed_text_blank)
            
            score = prediction[0][0] if isinstance(model, Sequential) else prediction[0]
            emotion_scores[emotion].append(score)
            if score >= 0.5:
                emotion_votes[emotion] += 1

        logging.debug(f"Using model type: BERT")
        # BERT Prediction
        if emotion in best_models_bert:
            bert_model = best_models_bert[emotion]
            with torch.no_grad():
                input_ids_tensor = torch.tensor(input_ids_bert).long()
                attention_mask_tensor = torch.tensor(attention_mask_bert).long()
                prediction = bert_model(input_ids_tensor, attention_mask_tensor)
                score = prediction.item()
                emotion_scores[emotion].append(score)
                if score >= 0.5:
                    emotion_votes[emotion] += 1

    # Log all scores and votes
    logging.debug(f"Emotion scores: {emotion_scores}")
    logging.debug(f"Emotion votes: {emotion_votes}")

    # Averaging scores
    average_scores = {emotion: np.mean(scores) for emotion, scores in emotion_scores.items()}
    detected_emotion_avg = max(average_scores, key=average_scores.get)
    logging.info(f"Detected emotion (average): {detected_emotion_avg}")

    # Majority voting
    detected_emotion_vote = max(emotion_votes, key=emotion_votes.get)
    logging.info(f"Detected emotion (vote): {detected_emotion_vote}")

    return detected_emotion_avg, average_scores, detected_emotion_vote, emotion_votes

# Evaluation
def evaluate_ensemble(X_test, y_test):
    logging.info("Evaluating ensemble model...")
    predictions_avg = []
    predictions_vote = []
    
    for text in X_test:
        detected_emotion_avg, _, detected_emotion_vote, _ = predict_emotion(text)
        predictions_avg.append(detected_emotion_avg)
        predictions_vote.append(detected_emotion_vote)
    
    # Calculate accuracy
    avg_accuracy = accuracy_score(y_test, predictions_avg)
    vote_accuracy = accuracy_score(y_test, predictions_vote)
    
    # Generate classification reports
    avg_classification_report = classification_report(y_test, predictions_avg, target_names=emotions)
    vote_classification_report = classification_report(y_test, predictions_vote, target_names=emotions)
    
    logging.info(f"Ensemble Model - Averaging Accuracy: {avg_accuracy}")
    logging.info(f"Ensemble Model - Voting Accuracy: {vote_accuracy}")
    logging.info("Ensemble Model - Averaging Classification Report:\n" + avg_classification_report)
    logging.info("Ensemble Model - Voting Classification Report:\n" + vote_classification_report)
    
    return avg_accuracy, vote_accuracy, avg_classification_report, vote_classification_report

# Load the test data from the CSV file
csv_path = '../data/arabic_sentiment.csv'

X_test, y_test = load_test_data(csv_path)

# Evaluate the ensemble model
avg_accuracy, vote_accuracy, avg_classification_report, vote_classification_report = evaluate_ensemble(X_test, y_test)

print(f"Ensemble Model - Averaging Accuracy: {avg_accuracy}")
print(f"Ensemble Model - Voting Accuracy: {vote_accuracy}")
print("Ensemble Model - Averaging Classification Report:\n", avg_classification_report)
print("Ensemble Model - Voting Classification Report:\n", vote_classification_report)

