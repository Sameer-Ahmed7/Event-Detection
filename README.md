# Project Title:
Event Detection

# Overview:
This repository is dedicated to the event detection task, which constitutes my **Homework 1** for the course **MULTILINGUAL NATURAL LANGUAGE PROCESSING**, part of my Master’s in AI and Robotics at [Sapienza University of Rome](https://www.uniroma1.it/it/pagina-strutturale/home). The project explores three different approaches to event detection:

* **Bidirectional LSTM (BiLSTM) without pre-trained word embeddings:** Examines the baseline capabilities of BiLSTM in event detection.
* **Bidirectional LSTM (BiLSTM) with pre-trained word embeddings:** Investigates the impact of leveraging pre-trained embeddings on the model's performance.
* **Bidirectional LSTM with Conditional Random Field (CRF):** Assesses the combined approach for improved event classification accuracy.

Each method is meticulously analyzed to determine its effectiveness in accurately identifying events within the text, providing a holistic view of current techniques in the field.

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
<img src="https://github.com/Sameer-Ahmed7/Event-Detection/blob/main/Images/event_detection.png" width="50%" height="50%" >
</p>



