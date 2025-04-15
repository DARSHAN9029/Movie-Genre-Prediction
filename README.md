# 🎬 Movie Genre Prediction using NLP & Machine Learning

This project is a Machine Learning model that predicts the **genre** of a movie based on its **title and description**. It uses **TF-IDF vectorization** for text features and a **Linear Support Vector Classifier (LinearSVC)** for classification.

---

## 📌 Features

- Input: Movie Title & Description
- Output: Predicted Genre
- Techniques Used:
  - Text Preprocessing & Cleaning
  - TF-IDF Vectorization
  - LinearSVC Classifier
  - Accuracy and Performance Metrics

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📂 Dataset

The dataset contains:
- `ID`: Movie ID
- `Title`: Movie title
- `Description`: Movie plot/summary
- `Genre`: Target label for classification

Split into:
- `train_data.txt`
- `test_data.txt`

---

## 🔍 Workflow

1. Load & clean data
2. Apply custom text cleaning function
3. TF-IDF vectorization
4. Train a `LinearSVC` model
5. Evaluate with accuracy and confusion matrix
6. Use model to predict genres from new movie descriptions

