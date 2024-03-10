# Project Title:
Event Detection



# Overview:
This repository is dedicated to the event detection task, which constitutes my **Homework 1** for the course **MULTILINGUAL NATURAL LANGUAGE PROCESSING**, part of my Master’s in AI and Robotics at [Sapienza University of Rome](https://www.uniroma1.it/it/pagina-strutturale/home). The project explores three different approaches to event detection:

1. **Bidirectional LSTM (BiLSTM) without pre-trained word embeddings:** Examines the baseline capabilities of BiLSTM in event detection.
2. **Bidirectional LSTM (BiLSTM) with pre-trained word embeddings:** Investigates the impact of leveraging pre-trained embeddings on the model's performance.
3. **Bidirectional LSTM with Conditional Random Field (CRF):** Assesses the combined approach for improved event classification accuracy.

Each method is meticulously analyzed to determine its effectiveness in accurately identifying events within the text, providing a holistic view of current techniques in the field.

> [!IMPORTANT]
> For Homework 2 on Coarse-Grained Word Sense Disambiguation (WSD) as part of my Multilingual NLP course, please visit the [Coarse-Grained Word Sense Disambiguation (WSD)](https://github.com/Sameer-Ahmed7/Coarse-Grained-WSD) Repository.

# What is Event Detection?

Event detection (ED) is a task in Natural Language Processing (NLP) that aims to identify event triggers and categorize events mentioned in the text. 

These event detection labels are based on the BIO format. The total number of labels according to BIO format is:
- B-Sentiment
- B-Scenario
- B-Change
- B-Possession
- B-Action
- I-Sentiment
- I-Scenario
- I-Change
- I-Possession
- I-Action
- O


<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Event-Detection/blob/main/assets/event_detection.png" width="70%" height="70%" title="Event Detection">
  </p>

# Flow Diagram of Model:

This flow diagram illustrates the complete process of the event detection task.
<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Event-Detection/blob/main/assets/model_flow.png" width="70%" height="70%" title="Flow Diagram of Event Detection Task">
  </p>

# Data Preprocessing Steps:
Our dataset is not in the form that we give to our model. We need to preprocess our training, testing, and development (Validation) data.
1. The first step is to convert (training, development, and testing) tokens into lowercase because the capitalized forms of the words will give different embeddings from the lowercase forms. 
2. Then convert (training, testing, and development) tokens and labels into numbers. These tokens and labels need to be converted by taking training data (tokens and labels) unique values.
3. All tokens and labels are not the same length. So, for that, we need to apply padding.

# Class Imbalance Problem:

Due to the predominance of the "O" label in our dataset, we have a significant class imbalance. This can skew the accuracy metric, leading to misleadingly high performance when the model simply predicts the majority class.

To address this, we evaluate our models using the _macro F1-score_ instead of _accuracy_. The macro F1-score considers both precision and recall for each class and then takes the average, treating all classes equally. This makes it a more suitable metric for datasets with imbalanced classes, ensuring that our models are evaluated fairly based on their ability to identify all labels accurately, not just the majority class.

<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Event-Detection/blob/main/assets/data_imbalance.png" width="70%" height="70%" title="Data Imbalance">
  </p>

# Model Approaches: 

In this task, I am considering three different approaches:
1. Bidirectional LSTM (BiLSTM) without pre-trained word embedding
2. Bidirectional LSTM (BiLSTM) with pre-trained word embedding
3. Bidirectional LSTM with Conditional random field (CRF)

## 1. Bidirectional LSTM (BiLSTM) without pre-trained word embedding:
### Implementation Details:

The model’s first approach utilizes a Bidirectional LSTM (BiLSTM) without pre-trained word embeddings. BiLSTM is advantageous as it processes data from both directions of the sequence, unlike traditional LSTM, enhancing the model’s ability to capture the sequential context of words and phrases. An embedding layer precedes the BiLSTM layer to facilitate the model’s understanding of linguistic relationships through word embeddings. Despite experimenting with various hyperparameters to improve the model’s F1 score, such as activation functions, optimizers, and dimensions of the embedding and BiLSTM layers, no significant increase in F1 score was observed. Consequently, the architecture was finalized as shown in the code snippet below. The model employs categorical cross-entropy loss with an ignore index for padding, suitable for predicting the dataset’s 11 distinct classes.

```python
# Model 1 (Without Pre-trained Word Embedding)

MODEL_1(
  (embedding_layer): Embedding(33081, 300, padding_idx=0)
  (lstm): LSTM(300, 128, batch_first=True, bidirectional=True)
  (dropout1): Dropout(p=0.4, inplace=False)
  (fc1): Linear(in_features=256, out_features=32, bias=True)
  (dropout2): Dropout(p=0.4, inplace=False)
  (fc2): Linear(in_features=32, out_features=12, bias=True)
  (softmax): Softmax(dim=1)
)
```

### Confusion Matrix:

<p align="center">
<img src="https://github.com/Sameer-Ahmed7/Event-Detection/blob/main/assets/model_1_confusion_matrix.png" width="70%" height="70%" title="Model-1 Confusion Matrix">
  </p>


















