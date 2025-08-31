# BERT English Sentence Analysis 📚
# 📖 Project Overview
This project fine-tunes a BERT model for binary sequence classification on English sentences to determine grammatical acceptability. Using the CoLA (Corpus of Linguistic Acceptability) dataset, it classifies sentences as "acceptable" (1) or "unacceptable" (0). The implementation leverages PyTorch and the pytorch_pretrained_bert library for model handling, training, and evaluation.
Ideal for NLP tasks like grammar checking, linguistic analysis, or educational tools.

# ✨ Features

Data Preprocessing: Tokenization with BERT tokenizer, padding, and attention masks.
Model Fine-Tuning: BERT for sequence classification with binary output.
Training Pipeline: GPU-accelerated training with Adam optimizer and cross-entropy loss.
Evaluation: Uses Matthews correlation coefficient for imbalanced binary classification.
Modular Code: Easy to adapt for other NLP classification tasks.


# 📊 Dataset
The model is trained on the CoLA (Corpus of Linguistic Acceptability) dataset.

Training Data: 'in_domain_train.tsv' (~8,551 sentences).
Validation Data: 'out_of_domain_dev.tsv' (~516 sentences).
Classes: Binary (0: unacceptable, 1: acceptable).
Label Distribution: Imbalanced; more acceptable sentences (exact split not detailed in notebook).
Preprocessing: Sentences tokenized to max length 128, with [CLS] and [SEP] tokens.
Usage Notes: Focuses on single-sentence grammaticality judgment.

Evaluation results: Matthews correlation ~0.437 on dev set after 4 epochs.

# 🏗️ Project Structure
textbert-english-sentence-analysis/

├── BERT_English_Sentence_Analysis.ipynb  # Data loading, preprocessing, training, and evaluation

├── data/

│   ├── in_domain_train.tsv               # Training data (not in repo; download from CoLA)

│   └── out_of_domain_dev.tsv             # Validation data (not in repo; download from CoLA)

├── README.md                             # Project documentation

└── LICENSE                               # MIT License

🔧 Design and Architecture
1. Data Preprocessing

Tokenization: Uses BERT tokenizer ('bert-base-uncased') to convert sentences to token IDs.
Padding: Sequences padded to max length 128.
Attention Masks: Binary masks to ignore padding tokens.
DataLoader: Batches data with random/sequential samplers for training/evaluation.

2. Model Architecture

BERT Model: BertForSequenceClassification (pre-trained 'bert-base-uncased', num_labels=2).
Fine-Tuning: Freezes base layers initially if needed; trains classification head.
Training Configuration:

Epochs: 4
Batch Size: 32
Optimizer: BertAdam (learning rate 2e-5, warmup 0.1)
Loss: Cross-entropy for binary classification



3. Evaluation

Metric: Matthews correlation coefficient (suitable for imbalanced data).
Prediction Workflow: Forward pass on validation batches, argmax on logits.

The design emphasizes efficiency with GPU support and standard BERT practices for NLP.

# 🚀 Installation

Install Dependencies:
bashpip install torch keras-preprocessing scikit-learn pytorch-pretrained-bert tqdm pandas numpy matplotlib

Download Dataset:

Obtain CoLA dataset from GitHub.
Place 'in_domain_train.tsv' and 'out_of_domain_dev.tsv' in the data/ directory.


GPU Support (optional but recommended):

Ensure CUDA is installed for faster training.


# 🛠️ Usage

Open BERT_English_Sentence_Analysis.ipynb in Jupyter Notebook/Colab.
Execute cells sequentially to:

Load and preprocess data.
Fine-tune the BERT model.
Evaluate on validation set (Matthews correlation printed).



Example:

Input Sentence: "The sailor dogs the telescope."
Output: 0 (unacceptable) or 1 (acceptable).

To classify new sentences:

Tokenize and pass through the trained model.


# 📋 Dependencies

Python 3.10+
PyTorch 1.x
pytorch-pretrained-bert
Keras Preprocessing
Scikit-learn
Tqdm
Pandas
NumPy
Matplotlib


# ⚠️ Limitations

Trained on English sentences only; may not handle other languages or dialects.
Binary classification; no nuance for partial grammaticality.
Evaluation on out-of-domain data may vary; fine-tune further for specific domains.
Uses older pytorch_pretrained_bert library (consider updating to Hugging Face Transformers).
