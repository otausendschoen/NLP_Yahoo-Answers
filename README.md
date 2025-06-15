# Few-Shot Text Classification with BERT and LLaMA-Augmented Data

This repository contains the full pipeline for an NLP classification project aimed at robust text categorization with minimal labeled data. The project evaluates several strategies for improving model performance in low-resource settings by leveraging both rule-based and large language model (LLM) techniques.


- Benchmark performance of **BERT-based** models under few-shot constraints
- Evaluate effectiveness of **synthetic data augmentation** using **LLaMA**
- Implement **zero-shot** and **semi-supervised learning** techniques
- Compare all approaches to state-of-the-art results using full datasets


### 1. Dataset & Baseline
- Chosen from the HuggingFace hub: (yahoo_answers_topics)
- Dataset statistics, random classifier expectations, and rule-based baseline implemented.

### 2. Data Scientist Challenge
- **32-shot BERT** classifier
- **Geometric and LLaMA-based data augmentation**
- **Zero-shot inference** using ChatGPT/LLaMA
- **Label generation** with LLaMA to enhance training set
- Comparison of each approach via metrics like F1-score, Accuracy

### 3. Full Dataset Training
- Learning curve across [1%, 10%, 25%, 50%, 75%, 100%] dataset splits
- Evaluation of hybrid training strategies combining synthetic + real data

### 4. Model Optimization
- Distillation and quantization of the best BERT model
- Inference speed and performance trade-off analysis



## Getting Started

```bash
git clone https://github.com/yourusername/few-shot-llama-classifier.git
cd few-shot-llama-classifier
pip install -r requirements.txt

## Exectutive Summary:
**Purpose and Scope **
This project explores strategies to enhance text classification performance in low-resource scenarios, where labeled data is limited. The goal is to develop a robust NLP pipeline by combining rule-based methods with large language model (LLM) techniques. The scope includes benchmarking BERT-based models under few-shot settings, testing synthetic data augmentation using LLaMA, and implementing zero-shot and semi-supervised learning approaches.
We used the Random Classifier as a bottom benchmark, where we obtained 0.1002 in the accuracy metrics.

**Findings**
- When trying to enhance the model with limited data, we observe that there is no significant difference between using a stratified sample of the 32 labeled examples and selecting them randomly. The results are very similar, likely because with such a small amount of data, the model cannot learn effectively. This may be due to the diversity in question types in terms of both structure and content, as the dataset is human-generated. With so few examples, the model struggles to generalize. In both the stratified and unstratified cases, performance is only slightly better than random classification—by approximately 0.14. 
