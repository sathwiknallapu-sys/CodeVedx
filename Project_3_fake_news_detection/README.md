# 📰 AI Based Fake News Detection Tool

## 📌 Project Overview

The **AI Based Fake News Detection Tool** is a Machine Learning-based web application that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP).

The application preprocesses news text, converts it into TF-IDF features, and uses a trained Machine Learning model to predict whether the news is genuine.

---

# 🚀 Features

- Fake / Real News Detection
- Confidence Score
- Text Preprocessing
- TF-IDF Vectorization
- Logistic Regression Model
- Model Saving (.pkl)
- Flask Web Interface
- Bootstrap UI
- Exception Handling
- Logging
- Modular Project Structure

---

# 🛠 Technologies Used

- Python 3.11
- Flask
- Scikit-Learn
- Pandas
- NumPy
- NLTK
- Joblib
- HTML
- CSS
- Bootstrap 5

---

# 📂 Project Structure

```
Project_3
│
├── app.py
├── train.py
├── predict.py
├── config.py
├── requirements.txt
│
├── dataset/
│     fake_news.csv
│
├── model/
│     fake_news_model.pkl
│     tfidf_vectorizer.pkl
│
├── preprocessing/
│
├── ml/
│
├── utils/
│
├── templates/
│
├── static/
│
└── README.md
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/YourUsername/CODEVEDX.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Train Model

```bash
python train.py
```

---

# ▶ Run Application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# Dataset

Dataset contains

- News Text
- Label

Label Values

```
0 → Real News

1 → Fake News
```

---

# Machine Learning Pipeline

```
Dataset

↓

Text Cleaning

↓

Stopword Removal

↓

Lemmatization

↓

TF-IDF

↓

Logistic Regression

↓

Prediction

↓

Confidence Score
```

---

# Sample Prediction

Input

```
NASA announced a successful satellite launch.
```

Output

```
Prediction : Real News

Confidence : 99%
```

---

# Future Improvements

- Deep Learning Models
- BERT Transformer
- News API Integration
- Multi-language Support
- User Authentication
- Cloud Deployment

---

# Author

**Sathwik Nallapu**

Artificial Intelligence & Machine Learning Intern

CODEVEDX
