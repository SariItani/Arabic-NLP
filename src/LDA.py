import pickle
import numpy as np
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Load preprocessed data and models
with open('../models/dad_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
embedding_matrix = np.load('../models/dad_embedding_matrix.npy')
lda_model = LdaModel.load('../models/lda_model.gensim')

# Load preprocessed emotion data
emotion_dfs = {}
emotions_list = [
    "شعور البغض - Hatred",
    "شعور الحب - Liking",
    "شعور الحزن – Sadness ", 
    "شعور الخوف – Fear",
    "شعور الفرح - Joy"
]
for emotion in emotions_list:
    tokenized_data = np.load(f'../data/data_{emotion}.npy')
    labels = np.load(f'../data/labels_{emotion}.npy')
    emotion_dfs[emotion] = (tokenized_data, labels)

# Function to get LDA features
def get_lda_features(texts, lda_model, dictionary):
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_features = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
        topic_vector = [prob for _, prob in topic_dist]
        lda_features.append(topic_vector)
    return np.array(lda_features)

# Initialize data for performance tracking
model_performance = []

# Define models to test
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC(probability=True)),
    ('MLP', MLPClassifier(max_iter=1000)),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('Bernoulli Naive Bayes', BernoulliNB())
]

model_metrics = {model_name: [] for model_name, _ in models}
model_metrics["CNN"] = []  # Initialize CNN key
model_metrics["LSTM"] = []  # Initialize LSTM key

# CNN and LSTM models
def build_cnn_model(embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], 
                        output_dim=embedding_matrix.shape[1], 
                        weights=[embedding_matrix], 
                        input_length=100,  # Ensure this matches the length used in pad_sequences
                        trainable=False))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def build_lstm_model(embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], 
                        output_dim=embedding_matrix.shape[1], 
                        weights=[embedding_matrix], 
                        input_length=100,  # Ensure this matches the length used in pad_sequences
                        trainable=False))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Train models for each emotion
for emotion in emotion_dfs.keys():
    X, y = emotion_dfs[emotion]
    
    # Negative samples (all other emotions)
    X_negative = np.concatenate([emotion_dfs[emo][0] for emo in emotion_dfs.keys() if emo != emotion])
    y_negative = np.zeros(X_negative.shape[0])
    
    # Create a balanced dataset by combining positive and negative samples
    X_combined = np.vstack([X, X_negative])
    y_combined = np.concatenate([y, y_negative])
    
    # Shuffle combined dataset
    indices = np.arange(X_combined.shape[0])
    np.random.shuffle(indices)
    X_combined, y_combined = X_combined[indices], y_combined[indices]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    best_score = 0
    best_auc = 0
    best_model = None
    all_model_metrics = []

    for model_name, model in models:
        try:
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            score = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics = report.get('0.0', {})
            f1 = metrics.get('f1-score', None)
            recall = metrics.get('recall', None)
            precision = metrics.get('precision', None)

            # Save accuracy and ROC AUC for each model-emotion pair
            model_metrics[model_name].append((emotion, score, auc, f1, recall, precision))

            # Save confusion matrix and ROC Curve as before
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10,7))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {emotion} - {model_name}')
            plt.savefig(f'../results/figures/{model_name}_{emotion}_confusion_matrix.png')
            plt.close()
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, marker='.')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {emotion} (Model: {model_name})')
            plt.savefig(f'../results/figures/{model_name}_{emotion}_roc_curve.png')
            plt.close()

            if score > best_score:
                best_score = score
                best_auc = auc
                best_f1 = f1
                best_recall = recall
                best_precision = precision
                best_model = model
                best_y_pred_prob = y_pred_prob
                
        except ValueError as e:
            print(f"Skipping {model_name} for {emotion}: {e}")

    # CNN and LSTM models
    X_train_padded = pad_sequences(X_train, maxlen=100)
    X_test_padded = pad_sequences(X_test, maxlen=100)

    print(f'X_train_padded shape: {X_train_padded.shape}')
    print(f'X_test_padded shape: {X_test_padded.shape}')

    # CNN Model
    cnn_model = build_cnn_model(embedding_matrix)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    cnn_model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_prob_cnn = cnn_model.predict(X_test_padded).ravel()
    y_pred_cnn = (y_pred_prob_cnn > 0.5).astype(int)
    cnn_score = accuracy_score(y_test, y_pred_cnn)
    cnn_auc = roc_auc_score(y_test, y_pred_prob_cnn)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = report.get('0.0', {})
    f1 = metrics.get('f1-score', None)
    recall = metrics.get('recall', None)
    precision = metrics.get('precision', None)
    
    # Save CNN accuracy and ROC AUC
    model_metrics["CNN"].append((emotion, score, auc, f1, recall, precision))

    # Save CNN confusion matrix and ROC Curve as before
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {emotion} - CNN')
    plt.savefig(f'../results/figures/CNN_{emotion}_confusion_matrix.png')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {emotion} (Model: CNN)')
    plt.savefig(f'../results/figures/CNN_{emotion}_roc_curve.png')
    plt.close()
    
    if score > best_score:
        best_score = score
        best_auc = auc
        best_f1 = f1
        best_recall = recall
        best_precision = precision
        best_model = ("CNN", cnn_model)
    
    # LSTM Model
    lstm_model = build_lstm_model(embedding_matrix)
    lstm_model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_prob_lstm = lstm_model.predict(X_test_padded).ravel()
    y_pred_lstm = (y_pred_prob_lstm > 0.5).astype(int)
    lstm_score = accuracy_score(y_test, y_pred_lstm)
    lstm_auc = roc_auc_score(y_test, y_pred_prob_lstm)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = report.get('0.0', {})
    f1 = metrics.get('f1-score', None)
    recall = metrics.get('recall', None)
    precision = metrics.get('precision', None)
    
    # Save LSTM accuracy and ROC AUC
    model_metrics["LSTM"].append((emotion, score, auc, f1, recall, precision))

    # Save LSTM confusion matrix and ROC Curve as before
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {emotion} - LSTM')
    plt.savefig(f'../results/figures/LSTM_{emotion}_confusion_matrix.png')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {emotion} (Model: LSTM)')
    plt.savefig(f'../results/figures/LSTM_{emotion}_roc_curve.png')
    plt.close()
    
    if score > best_score:
        best_score = score
        best_auc = auc
        best_f1 = f1
        best_recall = recall
        best_precision = precision
        best_model = ("LSTM", lstm_model)

    # Store the best model and its performance
    model_performance.append((f"{type(best_model).__name__} - {emotion}", best_score, best_auc, best_f1, best_recall, best_precision))

    # Save the best model for this emotion
    with open(f'../models/best_model_lda_{emotion}.pkl', 'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {emotion}')
    plt.savefig(f'../results/figures/lda_{emotion}_confusion_matrix.png')
    plt.close()

    # Save ROC Curve for the best model
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {emotion} (Best Model: {type(best_model).__name__})')
    plt.savefig(f'../results/figures/lda_{emotion}_roc_curve.png')
    plt.close()

    # Save evaluation report
    with open(f'../results/evaluation_reports/lda_{emotion}_report.txt', 'w') as f:
        f.write(f"Classification Report for {emotion}:\n")
        f.write(classification_report(y_test, best_model.predict(X_test)))
        for model_name, score, report, conf_matrix, auc in all_model_metrics:
            f.write(f"\nModel: {model_name}\n")
            f.write(f"Accuracy: {score:.4f}\n")
            f.write(f"ROC AUC: {auc:.4f}\n")
            f.write(f"Classification Report:\n{report}\n")

# Save the metrics to CSV files
for model_name, metrics in model_metrics.items():
    df_metrics = pd.DataFrame(metrics, columns=['Emotion', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'])
    df_metrics.to_csv(f'../results/evaluation_reports/lda_{model_name}_metrics.csv', index=False)

# Convert model performance data to DataFrame
df_performance = pd.DataFrame(model_performance, columns=['Model-Emotion', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'])

# Save performance summary
df_performance.to_csv('../results/performance_summary_lda.csv', index=False)

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x='Accuracy', y='Model-Emotion', data=df_performance, palette='viridis')
plt.xlabel('Accuracy')
plt.ylabel('Model-Emotion')
plt.title('Model Accuracy by Emotion for LDA')
plt.savefig('../results/figures/lda_model_accuracy.png')
plt.close()

# Plot AUC scores
plt.figure(figsize=(12, 8))
sns.barplot(x='AUC', y='Model-Emotion', data=df_performance, palette='viridis')
plt.xlabel('AUC')
plt.ylabel('Model-Emotion')
plt.title('Model AUC by Emotion for LDA')
plt.savefig('../results/figures/lda_model_auc.png')
plt.close()

# Print the overall performance summary
print("Model Performance Summary (LDA):")
print(df_performance)
