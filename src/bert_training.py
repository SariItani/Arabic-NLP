from model_definition import BERTClassifier
import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    # Load preprocessed BERT inputs and labels
    emotions = ["شعور الفرح - Joy", "شعور الحزن – Sadness ", "شعور الخوف – Fear", "شعور الحب - Liking", "شعور البغض - Hatred"]

    bert_inputs = {}
    bert_attention_masks = {}
    labels = {}

    for emotion in emotions:
        print("loading: " + emotion)
        bert_attention_masks[emotion] = torch.load(f'../data/bert_attention_mask_{emotion}.pt')
        bert_inputs[emotion] = torch.load(f'../data/bert_input_ids_{emotion}.pt')
        labels[emotion] = np.ones(bert_inputs[emotion].shape[0])  # Assuming positive class for the emotion

    # Initialize performance tracking
    model_performance = []

    # Training function
    def train_bert_model(model, train_dataloader, val_dataloader, epochs=5):
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        criterion = nn.BCELoss()
        
        # Track loss and accuracy
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            training_losses.append(avg_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, labels = batch
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            validation_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), training_losses, label='Training Loss')
        plt.plot(range(1, epochs + 1), validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('../results/figures/training_validation_loss.png')
        plt.close()

        # Validation predictions
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask)
                predictions.extend(outputs.squeeze().tolist())
                true_labels.extend(labels.tolist())

        return np.array(predictions), np.array(true_labels)

    # Train and evaluate the model for each emotion
    for emotion in emotions:
        # Create positive and negative datasets
        X_positive = bert_inputs[emotion]
        y_positive = labels[emotion]

        X_negative = torch.cat([bert_inputs[emo] for emo in emotions if emo != emotion])
        y_negative = np.zeros(X_negative.shape[0])

        # Combine positive and negative datasets
        X_combined = torch.cat((X_positive, X_negative))
        y_combined = np.concatenate((y_positive, y_negative))

        attention_mask_combined = torch.cat((bert_attention_masks[emotion], torch.cat([bert_attention_masks[emo] for emo in emotions if emo != emotion])))

        # Split into training and test sets
        X_train, X_test, y_train, y_test, attention_mask_train, attention_mask_test = train_test_split(
            X_combined, y_combined, attention_mask_combined, test_size=0.2, random_state=42
        )

        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            list(zip(X_train, attention_mask_train, y_train)), batch_size=16, shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            list(zip(X_test, attention_mask_test, y_test)), batch_size=16, shuffle=False
        )

        # Initialize model
        model = BERTClassifier()
        predictions, true_labels = train_bert_model(model, train_dataloader, val_dataloader)

        # Calculate metrics
        y_pred = (predictions > 0.5).astype(int)
        score = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, predictions)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Save the best model for this emotion
        os.makedirs('../models', exist_ok=True)
        torch.save(model.state_dict(), f'../models/best_model_bert_{emotion}.pt')

        # Save evaluation report
        os.makedirs('../results/evaluation_reports', exist_ok=True)
        with open(f'../results/evaluation_reports/bert_{emotion}_report.txt', 'w') as f:
            f.write(f"Classification Report for {emotion}:\n")
            f.write(classification_report(y_test, y_pred))
            f.write(f"\nAccuracy: {score:.4f}\n")
            f.write(f"ROC AUC: {auc:.4f}\n")

        # Save confusion matrix
        os.makedirs('../results/figures', exist_ok=True)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {emotion}')
        plt.savefig(f'../results/figures/bert_{emotion}_confusion_matrix.png')
        plt.close()

        # Save ROC Curve
        fpr, tpr, _ = roc_curve(y_test, predictions)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {emotion} (BERT)')
        plt.savefig(f'../results/figures/bert_{emotion}_roc_curve.png')
        plt.close()

        # Store performance metrics
        model_performance.append((f"BERT - {emotion}", score, auc))

    # Convert performance data to DataFrame
    df_performance = pd.DataFrame(model_performance, columns=['Model-Emotion', 'Accuracy', 'AUC'])

    # Plotting accuracy scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Accuracy', y='Model-Emotion', data=df_performance, palette='viridis')
    plt.xlabel('Accuracy')
    plt.ylabel('Model-Emotion')
    plt.title('Model Performance vs Emotion (BERT)')
    plt.tight_layout()
    plt.savefig('../results/figures/model_performance_vs_emotion_bert.png')
    plt.close()

    # Plotting AUC scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='AUC', y='Model-Emotion', data=df_performance, palette='viridis')
    plt.xlabel('AUC')
    plt.ylabel('Model-Emotion')
    plt.title('Model AUC vs Emotion (BERT)')
    plt.tight_layout()
    plt.savefig('../results/figures/model_auc_vs_emotion_bert.png')
    plt.close()

    # Save performance data to CSV
    df_performance.to_csv('../results/evaluation_reports/model_performance_bert.csv', index=False)

    # Print performance summary
    print("Model Performance Summary (BERT):")
    print(df_performance)

    # Print model architecture summary
    print("Model Architecture Summary:")
    print(model)  # Print model architecture summary
