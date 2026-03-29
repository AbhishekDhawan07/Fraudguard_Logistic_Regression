# 💳 Fraudguard Logistic Regression - Credit Card Fraud Detection using Logistic Regression

A supervised machine learning project that detects fraudulent credit card transactions using Logistic Regression. The project covers the complete ML pipeline -from exploratory data analysis, class imbalance handling via under-sampling, model training, and evaluation using both accuracy and precision scores.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset Description](#dataset-description)
4. [Project Workflow](#project-workflow)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [Model Performance](#model-performance)
9. [Key Insights](#key-insights)
10. [Author](#author)
11. [Contributing](#contributing)

---

## Project Overview

This project tackles one of the most critical real-world problems in financial security - detecting fraudulent credit card transactions. Using a highly imbalanced dataset of 284,807 transactions, the project applies **under-sampling** to handle class imbalance and trains a **Logistic Regression** classifier to distinguish legitimate transactions (Class = 0) from fraudulent ones (Class = 1).

The project demonstrates a clean end-to-end machine learning workflow in Python using `scikit-learn`, `pandas`, `matplotlib`, and `seaborn` — suitable as a portfolio project or a reference for handling imbalanced classification tasks.

---

## Features

- Full **EDA (Exploratory Data Analysis)** on a large-scale financial transactions dataset
- Null value detection and duplicate handling
- **Class distribution analysis** — frequency count, unique values, and percentage breakdown
- Comparative statistical analysis of legitimate vs. fraudulent transactions (`groupby().mean()`)
- **Under-Sampling** to address extreme class imbalance (99.83% legitimate vs. 0.17% fraud)
- Balanced dataset construction: 473 legitimate + 473 fraudulent transactions (946 total)
- **Stratified 80/20 train-test split** to preserve class ratio
- Logistic Regression model training
- Model evaluation using both **Accuracy Score** and **Precision Score** on training and test sets

---

## Dataset Description

| File | Description | Link |
|---|---|---|
| `creditcard.csv` | Credit card transaction records with PCA-transformed features and fraud labels | [📥 Download Dataset](https://drive.google.com/file/d/1lrUT-fwW1LNVCxsa_C2VKd8tY0wA4NnC/view?usp=sharing) |

**Dataset stats (after deduplication):**

- Total records: 284,807 (raw) → 283,726 (after removing 1,081 duplicates)
- Legitimate transactions (Class = 0): 283,253 (~99.83%)
- Fraudulent transactions (Class = 1): 473 (~0.17%)
- Memory usage: ~67.4 MB

**Column descriptions:**

| Column | Type | Description |
|---|---|---|
| `Time` | float64 | Seconds elapsed between a transaction and the first transaction in the dataset |
| `V1` – `V28` | float64 | PCA-transformed features - original transaction data anonymized to protect customer privacy |
| `Amount` | float64 | Transaction amount in currency units |
| `Class` | int64 | Target label: `0` = Legitimate, `1` = Fraudulent |

> **Input Features**: `Time`, `V1`–`V28`, `Amount`  
> **Output Feature**: `Class`

---

## Project Workflow

### 🟢 Step 1 — Data Loading & Overview
Load the dataset, inspect shape, column data types, null values, and overall structure.

```python
df = pd.read_csv("creditcard.csv")
df.shape        # (284807, 31)
df.info()
df.isnull().sum()
```

---

### 🔵 Step 2 — Exploratory Data Analysis (EDA)

**Missing Values:** No missing values found in any column across all 284,807 rows.

**Duplicate Handling:** 1,081 duplicate rows detected and removed.

**Class Distribution:**
```
Class 0 (Legitimate): 283,253  →  99.83%
Class 1 (Fraudulent):     473  →   0.17%
```
Observation: Data is **highly imbalanced** - fraud cases are extremely rare.

**Statistical Comparison (groupby mean):**

| Feature | Legitimate (Class 0) | Fraudulent (Class 1) |
|---|---|---|
| Time | ~94,835 sec | ~80,450 sec |
| Amount | ~$88.41 | ~$123.87 |
| V1 – V28 | Near 0 | Far from 0 (distinctive patterns) |

Key observations: Fraud transactions tend to occur earlier in the dataset, involve higher average amounts, and show strongly deviant PCA feature values - all of which are useful signals for the model.

---

### 🟡 Step 3 — Handling Class Imbalance (Under-Sampling)

**Why Under-Sampling over Oversampling?**
- The dataset is very large — removing some normal transactions loses minimal information
- Fraud data is real and rare — duplicating it (oversampling) risks overfitting
- Under-sampling keeps only real data and results in faster, more generalized training

**Under-Sampling approach:**
```python
legit_sample = legit.sample(n=473)   # Random sample of 473 legitimate transactions
new_dataset = pd.concat([legit_sample, fraud], axis=0)
# Balanced: 473 legitimate + 473 fraudulent = 946 total
```

Post-sampling class distribution: **Class 0: 473 | Class 1: 473** (perfectly balanced).

---

### 🟠 Step 4 — Feature & Target Split

```python
X = new_dataset.drop(columns='Class', axis=1)   # 30 input features
Y = new_dataset['Class']                          # Target label
# X.shape: (946, 30)
```

---

### 🔴 Step 5 — Train-Test Split (Stratified)

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
# X.shape: (946, 30) → X_train: (756, 30), X_test: (190, 30)
```

`stratify=Y` ensures both splits maintain the 50/50 class balance.

---

### 🟣 Step 6 — Model Training

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

### ⚫ Step 7 — Model Evaluation

Evaluated using **Accuracy Score** and **Precision Score** on both training and test data.

```python
# Accuracy
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Precision
training_data_precision = precision_score(Y_train, X_train_prediction)
test_data_precision = precision_score(Y_test, X_test_prediction)
```

---

## Tech Stack

- **Language**: Python 3.10
- **Libraries**:
  - `pandas` — data loading, manipulation, and cleaning
  - `numpy` — numerical operations
  - `matplotlib` & `seaborn` — data visualization
  - `scikit-learn` — Logistic Regression, train-test split, accuracy score, precision score
- **Environment**: Jupyter Notebook

---

## Project Structure

```
LogiSpamDetector/
│
├── README.md
│
└── Logistic Regression Project - Credit Card Fraud Detection/
    ├── Logistic Regression Project - Credit Card Fraud Detection.ipynb
    └── creditcard.csv
```

> 📥 The `creditcard.csv` dataset is not included in the repository due to its large size (~67.4 MB).  
> Download it here: [Google Drive Link](https://drive.google.com/file/d/1lrUT-fwW1LNVCxsa_C2VKd8tY0wA4NnC/view?usp=sharing)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/LogiSpamDetector.git
cd LogiSpamDetector
```

### 2. Download the dataset

Download `creditcard.csv` from the link above and place it in the project folder:

```
Logistic Regression Project - Credit Card Fraud Detection/creditcard.csv
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 4. Launch the notebook

```bash
jupyter notebook "Logistic Regression Project - Credit Card Fraud Detection.ipynb"
```

### 5. Run all cells in order

The notebook is fully self-contained. Run cells top to bottom to reproduce the full pipeline.

---

## Model Performance

| Metric | Training Set | Test Set |
|---|---|---|
| **Accuracy** | ~93.78% | ~92.63% |
| **Precision** | ~96.10% | ~94.51% |

**Accuracy interpretation:**
- Training: the model correctly predicted ~94 out of 100 transactions it was trained on
- Test: the model correctly predicted ~93 out of 100 unseen transactions

**Precision interpretation:**
- Training: out of 100 transactions flagged as fraud, ~96 were actually fraudulent (only ~4 false positives)
- Test: out of 100 transactions flagged as fraud, ~95 were actually fraudulent (only ~5 false positives)

Training and test metrics are close to each other, confirming the model is **neither overfitting nor underfitting** — a well-generalized classifier despite the extreme original class imbalance.

---

## Key Insights

- The dataset is **extremely imbalanced** (~99.83% legitimate, ~0.17% fraud) — a direct real-world reflection of how rare fraud is
- **1,081 duplicate rows** (~0.38% of data) were removed before analysis — an important cleaning step
- Fraudulent transactions have a **higher average transaction amount** (~$124 vs. ~$88 for legitimate) — a strong fraud signal
- PCA features `V1`–`V28` show **dramatically different mean values** for fraud vs. legitimate transactions — indicating these anonymized features carry strong discriminative power
- **Under-sampling** outperforms oversampling here: the dataset is large enough that losing some normal transactions is negligible, while duplicating fraud records risks overfitting
- High **precision (~94.5% on test)** means the model produces very few false alarms — important for a fraud detection system where unnecessary transaction blocks harm user experience

---

## Author

Built as a Python machine learning portfolio project to demonstrate handling of real-world class imbalance, Logistic Regression classification, and multi-metric model evaluation on a large-scale financial dataset.

---

🤝 Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request 🚀

---

> ⭐ If you found this project useful, consider starring the repository!

