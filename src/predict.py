import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Sequential
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from gensim.models.ldamodel import LdaModel
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from model_definition import BERTClassifier

# Load tokenizer and embedding matrix
with open('../models/dad_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
embedding_matrix = np.load('../models/dad_embedding_matrix.npy')

# Load vectorizers and LDA model
with open('../models/bow_vectorizer.pkl', 'rb') as handle:
    bow_vectorizer = pickle.load(handle)
with open('../models/tfidf_vectorizer.pkl', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)
lda_model = LdaModel.load('../models/lda_model.gensim')
dictionary = Dictionary.load('../models/lda_model.gensim.id2word')

word2vec_model = Word2Vec.load("../models/dad_word2vec.model")

emotions = [
    "شعور البغض - Hatred",
    "شعور الحب - Liking",
    "شعور الحزن – Sadness ", 
    "شعور الخوف – Fear",
    "شعور الفرح - Joy"
]

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
best_models_bert = {}
for emotion in ["شعور الفرح - Joy", "شعور الحزن – Sadness ", "شعور الخوف – Fear", "شعور الحب - Liking", "شعور البغض - Hatred"]:
    model_path = f'../models/best_model_bert_{emotion}.pt'
    if os.path.exists(model_path):
        model = BERTClassifier()  # Create an instance of your custom class
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        # Set the model to evaluation mode
        model.eval()
        
        best_models_bert[emotion] = model
    else:
        print(f"Model file not found for {emotion}: {model_path}")

maxlen = 100
embedding_dim = 300

# Expected number of features for LDA
expected_num_features_lda = 100  # Adjust this according to your LDA model's num_topics

# Load best models
best_models = {}
best_models_lda = {}
best_models_word2vec = {}
best_models_tfidf = {}
best_models_bow = {}

for emotion in emotions:
    try:
        best_models_lda[emotion] = load_model(f'../models/best_model_lda_{emotion}.h5')
        best_models_word2vec[emotion] = load_model(f'../models/best_model_word2vec_{emotion}.h5')
        best_models_tfidf[emotion] = load_model(f'../models/best_model_tfidf_{emotion}.h5')
        best_models_bow[emotion] = load_model(f'../models/best_model_bow_{emotion}.h5')
        best_models[emotion] = load_model(f'../models/best_model_{emotion}.h5')
    except:
        with open(f'../models/best_model_lda_{emotion}.pkl', 'rb') as handle:
            best_models_lda[emotion] = pickle.load(handle)
        with open(f'../models/best_model_word2vec_{emotion}.pkl', 'rb') as handle:
            best_models_word2vec[emotion] = pickle.load(handle)
        with open(f'../models/best_model_tfidf_{emotion}.pkl', 'rb') as handle:
            best_models_tfidf[emotion] = pickle.load(handle)
        with open(f'../models/best_model_bow_{emotion}.pkl', 'rb') as handle:
            best_models_bow[emotion] = pickle.load(handle)
        with open(f'../models/best_model_{emotion}.pkl', 'rb') as handle:
            best_models[emotion] = pickle.load(handle)

def preprocess_text(text, maxlen):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

def preprocess_text_lda(text, num_topics):
    tokens = text.split()
    bow = dictionary.doc2bow(tokens)
    lda_features = [0] * num_topics
    for idx, prob in lda_model.get_document_topics(bow, minimum_probability=0):
        lda_features[idx] = prob
    # Pad or truncate the features to ensure it matches the expected number of topics
    if len(lda_features) < num_topics:
        lda_features.extend([0] * (num_topics - len(lda_features)))
    elif len(lda_features) > num_topics:
        lda_features = lda_features[:num_topics]
    return np.array([lda_features])

def preprocess_text_tfidf(text):
    return tfidf_vectorizer.transform([text]).toarray()

def preprocess_text_bow(text):
    return bow_vectorizer.transform([text]).toarray()

def preprocess_text_word2vec(text, model, embedding_dim):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.array([np.mean(word_vectors, axis=0)])
    else:
        return np.zeros((1, embedding_dim))

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

# Define the function for prediction
def predict_emotion(text):
    # Preprocess the text for each model type
    processed_text_blank = preprocess_text(text, maxlen)
    processed_text_lda = preprocess_text_lda(text, expected_num_features_lda)
    processed_text_tfidf = preprocess_text_tfidf(text)
    processed_text_bow = preprocess_text_bow(text)
    processed_text_word2vec = preprocess_text_word2vec(text, word2vec_model, embedding_dim)
    input_ids_bert, attention_mask_bert = preprocess_text_bert(text, bert_tokenizer, maxlen)

    # Print shapes for debugging
    print("Shapes of processed texts:")
    print(f"Blank/Word2Vec: {processed_text_blank.shape}")
    print(f"LDA: {processed_text_lda.shape}")
    print(f"TF-IDF: {processed_text_tfidf.shape}")
    print(f"BOW: {processed_text_bow.shape}")
    print(f"Word2Vec: {processed_text_word2vec.shape}")
    print(f"BERT: {input_ids_bert.shape}, {attention_mask_bert.shape}")

    # Initialize results for each model type
    emotion_scores = {}
    emotion_scores_lda = {}
    emotion_scores_word2vec = {}
    emotion_scores_tfidf = {}
    emotion_scores_bow = {}
    emotion_scores_bert = {}

    # Predict using blank models
    for emotion, model in best_models.items():
        prediction = model.predict(processed_text_blank)
        emotion_scores[emotion] = prediction[0][0] if isinstance(model, Sequential) else prediction[0]

    # Predict using LDA models
    for emotion, model in best_models_lda.items():
        prediction = model.predict(processed_text_lda)
        emotion_scores_lda[emotion] = prediction[0][0] if isinstance(model, Sequential) else prediction[0]

    # Predict using Word2Vec models
    for emotion, model in best_models_word2vec.items():
        prediction = model.predict(processed_text_word2vec)
        emotion_scores_word2vec[emotion] = prediction[0][0] if isinstance(model, Sequential) else prediction[0]

    # Predict using TF-IDF models
    for emotion, model in best_models_tfidf.items():
        prediction = model.predict(processed_text_tfidf)
        emotion_scores_tfidf[emotion] = prediction[0][0] if isinstance(model, Sequential) else prediction[0]

    # Predict using BOW models
    for emotion, model in best_models_bow.items():
        prediction = model.predict(processed_text_bow)
        emotion_scores_bow[emotion] = prediction[0][0] if isinstance(model, Sequential) else prediction[0]

    # Predict using BERT models
    for emotion, model in best_models_bert.items():
        # Convert NumPy arrays to PyTorch tensors
        input_ids_tensor = torch.tensor(input_ids_bert).long()
        attention_mask_tensor = torch.tensor(attention_mask_bert).long()
        
        # Make sure the model is in evaluation mode
        model.eval()
        
        # Use torch.no_grad() to disable gradient calculation
        with torch.no_grad():
            prediction = model(input_ids_tensor, attention_mask_tensor)
        
        # Convert the prediction to a Python float
        emotion_scores_bert[emotion] = prediction.item()

    # Get the most likely emotion for each model type
    detected_emotion = max(emotion_scores, key=emotion_scores.get)
    detected_emotion_lda = max(emotion_scores_lda, key=emotion_scores_lda.get)
    detected_emotion_word2vec = max(emotion_scores_word2vec, key=emotion_scores_word2vec.get)
    detected_emotion_tfidf = max(emotion_scores_tfidf, key=emotion_scores_tfidf.get)
    detected_emotion_bow = max(emotion_scores_bow, key=emotion_scores_bow.get)
    detected_emotion_bert = max(emotion_scores_bert, key=emotion_scores_bert.get)

    return (detected_emotion, emotion_scores, 
            detected_emotion_lda, emotion_scores_lda, 
            detected_emotion_word2vec, emotion_scores_word2vec, 
            detected_emotion_tfidf, emotion_scores_tfidf, 
            detected_emotion_bow, emotion_scores_bow,
            detected_emotion_bert, emotion_scores_bert)

text = input("أدخل النص العربي هنا: ")
(detected_emotion, emotion_scores, 
 detected_emotion_lda, emotion_scores_lda, 
 detected_emotion_word2vec, emotion_scores_word2vec, 
 detected_emotion_tfidf, emotion_scores_tfidf, 
 detected_emotion_bow, emotion_scores_bow,
 detected_emotion_bert, emotion_scores_bert) = predict_emotion(text)

print(f"Blank model detected Emotion: {detected_emotion}")
print("Blank model Scores:")
for emotion, score in emotion_scores.items():
    print(f"{emotion}: {score:.2f}")

print(f"LDA model detected Emotion: {detected_emotion_lda}")
print("LDA model Scores:")
for emotion, score in emotion_scores_lda.items():
    print(f"{emotion}: {score:.2f}")

print(f"Word2Vec model detected Emotion: {detected_emotion_word2vec}")
print("Word2Vec model Scores:")
for emotion, score in emotion_scores_word2vec.items():
    print(f"{emotion}: {score:.2f}")

print(f"TF-IDF model detected Emotion: {detected_emotion_tfidf}")
print("TF-IDF model Scores:")
for emotion, score in emotion_scores_tfidf.items():
    print(f"{emotion}: {score:.2f}")

print(f"BOW model detected Emotion: {detected_emotion_bow}")
print("BOW model Scores:")
for emotion, score in emotion_scores_bow.items():
    print(f"{emotion}: {score:.2f}")

print(f"BERT model detected Emotion: {detected_emotion_bert}")
print("BERT model Scores:")
for emotion, score in emotion_scores_bert.items():
    print(f"{emotion}: {score:.2f}")
