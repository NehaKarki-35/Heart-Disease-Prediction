# ❤️ Heart Disease Prediction

A machine learning classification project that compares multiple algorithms for predicting heart disease from patient health-related features, with an interactive Streamlit prediction interface.

## Overview

This project demonstrates an end-to-end machine learning workflow for a binary classification problem using a heart disease dataset.

The project covers:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Categorical feature encoding
* Feature scaling
* Stratified train-test splitting
* Comparison of five classification algorithms
* Model evaluation using Accuracy and F1 Score
* Saving the selected model and preprocessing objects
* Interactive prediction through a Streamlit application

> **Note:** This project is intended for educational and portfolio purposes. It is not a clinical diagnostic system.

---

## ✨ Key Features

* Exploratory Data Analysis
* Data inspection and descriptive statistics
* Duplicate checking
* Categorical feature encoding using one-hot encoding
* Feature scaling with `StandardScaler`
* Stratified 80/20 train-test split
* Comparison of 5 classification algorithms
* Accuracy and F1 Score evaluation
* KNN model selection based on test performance
* Serialized model and preprocessing artifacts
* Interactive Streamlit prediction interface

---

## 🛠️ Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* Decision Tree
* Support Vector Machine (SVM)
* StandardScaler

### Model Persistence

* Joblib

### Application

* Streamlit

---

## 📊 Dataset

The project uses a heart disease dataset containing **918 records and 12 original columns**.

The target variable is `HeartDisease`, while the remaining columns are used as input features.

### Dataset Features

| Feature          | Description                   |
| ---------------- | ----------------------------- |
| `Age`            | Patient age                   |
| `Sex`            | Patient sex                   |
| `ChestPainType`  | Chest pain category           |
| `RestingBP`      | Resting blood pressure        |
| `Cholesterol`    | Cholesterol level             |
| `FastingBS`      | Fasting blood sugar indicator |
| `RestingECG`     | Resting ECG category          |
| `MaxHR`          | Maximum heart rate            |
| `ExerciseAngina` | Exercise-induced angina       |
| `Oldpeak`        | ST depression                 |
| `ST_Slope`       | ST segment slope              |
| `HeartDisease`   | Target variable               |

The dataset contains a mixture of numerical and categorical variables.

---

## 🔄 Project Workflow

```text
Data Loading
     ↓
Data Inspection
     ↓
Exploratory Data Analysis
     ↓
Data Preprocessing
     ↓
Categorical Encoding
     ↓
Stratified Train/Test Split
     ↓
Standard Scaling
     ↓
Model Training
     ↓
Model Comparison
     ↓
KNN Selection
     ↓
Model Serialization
     ↓
Streamlit Prediction App
```

---

## 🔎 Exploratory Data Analysis

The notebook includes several steps to understand and prepare the dataset:

* Dataset shape inspection
* Data type analysis
* Descriptive statistics
* Duplicate checking
* Numerical feature analysis
* Categorical feature analysis
* Feature and target exploration
* Visualization of dataset characteristics

The EDA was performed before model training to understand the structure of the dataset and prepare the features for machine learning.

---

## ⚙️ Data Preprocessing

### Categorical Encoding

Categorical variables were converted into numerical features using:

```python
pd.get_dummies(df, drop_first=True)
```

This converts categorical values into dummy variables while dropping the first category to avoid redundant encoded columns.

### Train-Test Split

The dataset was divided using a stratified train-test split:

* **80%** training data
* **20%** testing data
* `random_state=42`
* Stratification based on the target variable

### Feature Scaling

`StandardScaler` was used to scale the features.

The scaler is fitted using the training data and then applied to the test data.

---

## 🤖 Machine Learning Models

Five classification algorithms were evaluated:

| Model                     |   Accuracy |   F1 Score |
| ------------------------- | ---------: | ---------: |
| Logistic Regression       |     87.50% |     0.8878 |
| K-Nearest Neighbors (KNN) | **88.59%** | **0.8986** |
| Gaussian Naive Bayes      |     86.96% |     0.8788 |
| Decision Tree             |     77.17% |     0.7835 |
| SVM (RBF Kernel)          |     86.41% |     0.8804 |

### Best Performing Model

**K-Nearest Neighbors (KNN)** achieved the highest Accuracy and F1 Score among the models tested in this experiment.

* **Accuracy:** 88.59%
* **F1 Score:** 0.8986

---

## 📈 Model Evaluation

Two primary metrics were used to compare the classification models.

### Accuracy

Accuracy measures the proportion of predictions that were classified correctly.

### F1 Score

F1 Score combines Precision and Recall into a single metric and is useful when evaluating classification performance beyond accuracy alone.

Comparing multiple algorithms helps identify which model performs best on the selected train-test split.

> The reported results are based on the project's stratified train-test evaluation and should not be interpreted as clinical performance.

---

## 🏆 Results

| Metric           |     Result |
| ---------------- | ---------: |
| Dataset Records  |        918 |
| Models Evaluated |          5 |
| Best Model       |        KNN |
| Test Accuracy    | **88.59%** |
| Test F1 Score    | **0.8986** |

The KNN model produced the strongest results among the five tested classification algorithms based on both Accuracy and F1 Score.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application through `app.py`.

The application loads the saved:

* KNN model
* Feature scaler
* Expected training columns

### User Inputs

The application accepts:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* ST Slope

### Prediction Flow

```text
User Input
    ↓
Create Input DataFrame
    ↓
Align Features with Training Columns
    ↓
Apply Saved StandardScaler
    ↓
Load Saved KNN Model
    ↓
Generate Prediction
    ↓
Display Result
```

The application displays either:

* **High Risk of Heart Disease**
* **Low Risk of Heart Disease**

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── heart.csv
├── heart_disease_analysis.ipynb
├── heart_columns.pkl
├── heart_scaler.pkl
├── knn_heart_model.pkl
├── .gitattributes
└── .ipynb_checkpoints/
```

### Important Files

| File                           | Purpose                                           |
| ------------------------------ | ------------------------------------------------- |
| `app.py`                       | Streamlit prediction application                  |
| `heart.csv`                    | Dataset                                           |
| `heart_disease_analysis.ipynb` | EDA, preprocessing, model training and evaluation |
| `knn_heart_model.pkl`          | Saved KNN model used by the application           |
| `heart_scaler.pkl`             | Saved feature scaler                              |
| `heart_columns.pkl`            | Saved expected feature columns                    |

---

## 🚀 Installation

This repository currently does not contain a `requirements.txt` file, so dependencies need to be installed manually.

### 1. Clone the Repository

```bash
git clone https://github.com/NehaKarki-35/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter
```

---

## ▶️ Run the Streamlit Application

After installing the dependencies, run:

```bash
streamlit run app.py
```

The Streamlit interface will open in your browser.

Enter the requested feature values and click **Predict** to generate a model prediction.

---

## 📓 Run the Notebook

The machine learning workflow is documented in:

```text
heart_disease_analysis.ipynb
```

Open the notebook using Jupyter Notebook or JupyterLab and execute the cells to explore the dataset, preprocessing steps, model training and evaluation.

---

## 🔮 Future Improvements

The following are potential improvements for future versions of the project:

* Hyperparameter tuning
* Cross-validation
* ROC-AUC evaluation
* Confusion matrix visualization
* Additional classification metrics
* Improved input validation
* Adding a `requirements.txt` file
* Automated testing
* Deployment configuration

These are future improvements and are not currently presented as implemented features.

---

## ⚠️ Medical Disclaimer

This project is intended for educational and portfolio purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 👩‍💻 Author

**Neha Karki**

GitHub: [NehaKarki-35](https://github.com/NehaKarki-35?utm_source=chatgpt.com)
