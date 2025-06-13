# 🛳️ Titanic Survival Prediction - ML Pipeline Project

This project demonstrates a complete end-to-end machine learning pipeline for predicting Titanic passenger survival using scikit-learn. It includes custom feature engineering, pipeline creation, hyperparameter tuning with `GridSearchCV`, and model interpretability via feature and permutation importances.

---

## 📌 Problem Statement

Predict whether a passenger survived the Titanic disaster using demographic and voyage data.

Dataset: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

---

## 🧠 Features Engineered

- `Title` — extracted from passenger name
- `FamilySize` — total family members on board
- `IsAlone` — binary feature (1 if alone, 0 otherwise)
- `Solo` — product of `IsAlone` and `Fare`
- `femClass` — interaction between `Sex` and `Pclass`

Removed columns: `Cabin`, `Ticket`, `PassengerId`

---

## 🏗️ Pipeline Components

Built using `sklearn.pipeline.Pipeline`, the components include:

- **Custom Transformers**: for title extraction and family-based features
- **Numerical Pipeline**: mean imputation + standard scaling
- **Categorical Pipeline**: mode imputation + one-hot encoding
- **ColumnTransformer**: applies different pipelines to numeric and categorical data
- **RandomForestClassifier**: final model with GridSearchCV

---

## 🔍 Model Selection & Evaluation

- **Best Estimator** selected using 5-fold `GridSearchCV`
- **Cross-validated** using `StratifiedKFold` with 10 splits
- **Accuracy**: ~83.7%

---

## 📊 Visual Outputs

Includes:
- **Feature Importance (from Random Forest)**
- **Permutation Importance (model-agnostic)**

Both are plotted using matplotlib.

---

## 🧪 How to Run This Project

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/titanic-ml-pipeline.git
cd titanic-ml-pipeline
```

2. **(Optional) Create a virtual environment**
```bash
  python -m venv venv
  source venv/bin/activate       # For Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
    pip install -r requirements.txt
```

4. **Add dataset**
   Download train.csv from Kaggle Titanic Competition and place it inside the data/ folder.

5. **Run the notebook**
```bash
  jupyter notebook titanic_pipeline.ipynb
```

## 🧰 Tech Stack
Python 3.x

pandas, numpy

scikit-learn

matplotlib

## 📘 Learnings
Through this project, I learned:

- How to create custom feature transformers using scikit-learn’s API

- How to structure a clean, modular ML pipeline

- How to perform hyperparameter tuning with GridSearchCV

- How to interpret model predictions using feature importance techniques

- How to make my first ML project GitHub-ready

## 🚀 Future Improvements
- Add test predictions and leaderboard submission logic

- Export model using joblib for reuse

- Deploy as API using Flask or FastAPI

## 🔗 Links
📊 Dataset: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

💼 Author: Your Name – LinkedIn
