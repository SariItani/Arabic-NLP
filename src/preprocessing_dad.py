import random
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import numpy as np
import pickle
from stop_words import get_stop_words
import string
import torch
from sklearn.model_selection import train_test_split

arabic_punctuation = "،؟؛!؛:()[]{}«»؛"

# Load the CSV file
df = pd.read_csv('../data/arabic_sentiment.csv')

# Create a dictionary to hold separate DataFrames for each emotion
emotion_dfs = {}
for column in df.columns[1:]:
    emotion_dfs[column] = df[['Unnamed: 0', column]].dropna().reset_index(drop=True)

# Download NLTK stop words
nltk.download('stopwords')
arabic_stop_words = set(stopwords.words('arabic'))
additional_stop_words = set(get_stop_words('ar'))
all_stop_words = arabic_stop_words.union(additional_stop_words)

with open("../data/stopwords.txt", 'w') as file:
    for word in all_stop_words:
        file.write(word + "\n")

with open("../data/nltk_stopwords.txt", 'w') as file:
    for word in arabic_stop_words:
        file.write(word + "\n")

with open("../data/other_stopwords.txt", 'w') as file:
    for word in additional_stop_words:
        file.write(word + "\n")

# Function to normalize Arabic text
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[ًٌٍَُِْ]", "", text)
    return text

# Function to remove stop words
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in all_stop_words])

def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation and char not in arabic_punctuation])

# Function to generate negation form of a sentence in Arabic
def negate_sentence_arabic(text):
    words = text.split()
    negation_words = ["لا", "لم", "لن", "ما"]
    if len(words) > 1:
        for i, word in enumerate(words):
            if word in ["هو", "هي", "هم", "هن", "أكون", "يكون", "تكون", "نكون", "كان", "كانت", "يكونوا", "نكونوا"]:
                words.insert(i + 1, random.choice(negation_words))
                break
        else:
            words.insert(0, random.choice(negation_words))
    else:
        words.insert(0, random.choice(negation_words))
    return ' '.join(words)

# Preprocess each emotion DataFrame separately and augment data with negations
augmented_emotion_dfs = {}
for emotion, df_emotion in emotion_dfs.items():
    df_emotion.iloc[:, 1] = df_emotion.iloc[:, 1].apply(normalize_arabic)
    df_emotion.iloc[:, 1] = df_emotion.iloc[:, 1].apply(remove_stopwords)
    df_emotion.iloc[:, 1] = df_emotion.iloc[:, 1].apply(remove_punctuation)
    
    # Generate negations
    df_negated = df_emotion.copy()
    df_negated.iloc[:, 1] = df_negated.iloc[:, 1].apply(negate_sentence_arabic)
    
    # Combine original and negated data
    augmented_emotion_dfs[emotion] = pd.concat([df_emotion, df_negated]).reset_index(drop=True)

# Display the augmented DataFrames
for emotion, df_emotion in augmented_emotion_dfs.items():
    print(f"Augmented DataFrame for {emotion}:\n", df_emotion.head())

# Initialize TF tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenized_data = {}
maxlen = 100

# Load pre-trained BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')

for emotion, df_emotion in augmented_emotion_dfs.items():
    texts = df_emotion.iloc[:, 1].values
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    tokenized_data[emotion] = padded_sequences

# Save the tokenizer
with open('../models/dad_tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train Word2Vec model
all_text = []
for emotion, df_emotion in augmented_emotion_dfs.items():
    all_text.extend(df_emotion.iloc[:, 1].values)

sentences = [text.split() for text in all_text]
word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
word2vec_model.save("../models/dad_word2vec.model")

# Create the embedding matrix
embedding_dim = 300
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# Save the embedding matrix
np.save('../models/dad_embedding_matrix.npy', embedding_matrix)

# Prepare and save labels
for emotion, df_emotion in augmented_emotion_dfs.items():
    labels = df_emotion.iloc[:, 1].apply(lambda x: 1 if x != '' else 0).values
    np.save(f'../data/labels_{emotion}.npy', labels)

# Save the processed data for each emotion
for emotion, data in tokenized_data.items():
    np.save(f'../data/data_{emotion}.npy', data)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
corpus = [df_emotion.iloc[:, 1].str.cat(sep=' ') for df_emotion in augmented_emotion_dfs.values()]
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
pickle.dump(tfidf_vectorizer, open('../models/tfidf_vectorizer.pkl', 'wb'))
np.save('../data/tfidf_matrix.npy', tfidf_matrix.toarray())

# LDA Model Training
dictionary = Dictionary([text.split() for text in corpus])
corpus_gensim = [dictionary.doc2bow(text.split()) for text in corpus]
lda_model = LdaModel(corpus_gensim, num_topics=10, id2word=dictionary, passes=15)
lda_model.save('../models/lda_model.gensim')

# Bag of Words Vectorization
bow_vectorizer = CountVectorizer(max_features=5000)
bow_matrix = bow_vectorizer.fit_transform(corpus)
pickle.dump(bow_vectorizer, open('../models/bow_vectorizer.pkl', 'wb'))
np.save('../data/bow_matrix.npy', bow_matrix.toarray())

print("Preprocessing, TF-IDF, LDA, and BOW model training complete.")

# Tokenize and encode the texts for each emotion
encoded_data = {}
maxlen = 100

for emotion, df_emotion in emotion_dfs.items():
    texts = df_emotion.iloc[:, 1].values
    encoded_texts = bert_tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=maxlen,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    encoded_data[emotion] = encoded_texts

# Optionally, split your data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the tokenized data and labels
for emotion, data in encoded_data.items():
    torch.save(data['input_ids'], f'../data/bert_input_ids_{emotion}.pt')
    torch.save(data['attention_mask'], f'../data/bert_attention_mask_{emotion}.pt')

print("BERT-specific preprocessing for Arabic complete.")