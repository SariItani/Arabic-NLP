# Arabic-NLP
 This paper presents a comprehensive study on sentiment analysis in the Arabic language, focusing on five categorical emotional dimensions. We introduce a novel Arabic corpus specifically designed for advanced sentiment analysis. We detail the specifics of a data augmentation technique that inverts the logic of sentences which reverses the predicted emotion, a detailed comparison of state-of-the-art NLP techniques such as TF-IDF, LDA, Bag of Words, etc. We trained machine learning (ML) and deep learning (DL) models for one-vs-many emotion classification and explored the effectiveness of an ensemble model combining these individual classifiers which scored high accuracy values. Furthermore, we analyze unjust voting system inherent in ensemble methods. Our approach highlights the challenges and opportunities in sentiment analysis, and our experimental framework encompasses data preprocessing, exploratory data analysis (EDA), and model training pipelining.

# How to run
* in ./src/ :
python3 preprocessing_dad.py && \
python3 eda_dad.py && \
python3 LDA.py && \    
python3 TF-IDF.py && \
python3 word2vec.py && \
python3 TensorflowTokenizer.py && \
python3 bagging.py && \
python3 bert_training.py && \
python3 ensemble.py

then,

python3 predict.py