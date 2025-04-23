# Mini Project 2 - Sentiment Analysis on Financial News

## Background

This project investigates the use of deep learning for sentiment analysis on financial news, focusing on transformer-based architectures like BERT and FinBERT. Leveraging both pre-trained and fine-tuned models, the team evaluates sentiment classification performance using financial news articles paired with corresponding stock return data. A dual-model approach was implemented, combining FinBERT for contextual understanding and VADER for lexical scoring. Multiple deep learning pipelines were explored, including fully connected neural networks (FCNN), Long Short-Term Memory (LSTM) networks, and a hybrid FinBERT-LSTM architecture. Fine-tuning was performed with hyperparameter variations across several BERT configurations, and performance was evaluated using metrics such as MSE, R², accuracy, and F1-score.

The FinBERT-LSTM hybrid achieved the strongest results, outperforming standalone MLP and LSTM models with an accuracy of 99.14% and an R² of 0.9998. Among fine-tuned BERT models, the BERT-large-uncased (Model D) offered the best overall performance balance. Interestingly, sentiment was found to be negatively correlated with stock returns during the test period, reflecting broader market uncertainty during the COVID-19 pandemic. This highlights the importance of contextual financial modeling. The findings point toward the value of integrating textual sentiment into financial forecasting, while also noting limitations such as sparse coverage for less prominent firms and the complexity of real-world financial sentiment dynamics.


### Installation

Required libraries are described in the requirements.txt file. The code should run with no issues using Python version 3.11+. Create a virutal environment of your choice. This project uses Miniconda:

```
conda create -n FinBERT python=3.11.7 jupyter
conda activate FinBERT
pip install -r requirements.txt
```