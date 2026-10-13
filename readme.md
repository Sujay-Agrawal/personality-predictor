# 🧠 Personality Type Prediction using Logistic Regression

Predicting whether a person is an **Introvert**, **Ambivert**, or **Extrovert** from 29 psychological, social, and behavioral traits — built with Python and scikit-learn.

The project includes full Exploratory Data Analysis (EDA), one-way ANOVA for statistical feature selection, and a trained Logistic Regression model that reaches **99.75% test accuracy**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.75%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Understanding a person's personality type can help improve recommendation systems, hiring tools, mental-health apps, and social platforms.

This project builds a machine learning model that predicts **personality type** from behavioral traits such as social energy, talkativeness, deep reflection, party liking, empathy, and leadership.

---

## 📂 Dataset

- **File:** `personality_synthetic_dataset.csv`
- **Size:** 20,000 rows × 30 columns
- **Target variable:** `personality_type` (Introvert / Ambivert / Extrovert)
- **Missing values:** None
- **Duplicates:** None

### Feature Overview

| Feature | Type | Description |
| :--- | :---: | :--- |
| `personality_type` | Categorical | **Target:** Introvert / Ambivert / Extrovert |
| `social_energy` | Numeric | Energy from social interaction |
| `alone_time_preference` | Numeric | Preference for being alone |
| `talkativeness` | Numeric | How talkative the person is |
| `deep_reflection` | Numeric | Tendency to self-reflect deeply |
| `group_comfort` | Numeric | Comfort level in groups |
| `party_liking` | Numeric | Enjoyment of parties |
| `listening_skill` | Numeric | Ability to listen |
| `empathy` | Numeric | Emotional empathy level |
| `creativity` | Numeric | Creative thinking |
| `organization` | Numeric | Organizational habits |
| `leadership` | Numeric | Leadership tendency |
| `risk_taking` | Numeric | Willingness to take risks |
| `public_speaking_comfort` | Numeric | Comfort with public speaking |
| `curiosity` | Numeric | Curiosity level |
| `routine_preference` | Numeric | Preference for routine |
| `excitement_seeking` | Numeric | Thrill-seeking behavior |
| `friendliness` | Numeric | Friendliness score |
| `emotional_stability` | Numeric | Emotional stability |
| `planning` | Numeric | Planning habits |
| `spontaneity` | Numeric | Spontaneous behavior |
| `adventurousness` | Numeric | Adventurousness |
| `reading_habit` | Numeric | Reading frequency |
| `sports_interest` | Numeric | Interest in sports |
| `online_social_usage` | Numeric | Online social media usage |
| `travel_desire` | Numeric | Desire to travel |
| `gadget_usage` | Numeric | Gadget usage |
| `work_style_collaborative` | Numeric | Collaborative work style |
| `decision_speed` | Numeric | Speed of decision-making |
| `stress_handling` | Numeric | Stress handling ability |

---

## 🔧 Project Workflow

1. **Data exploration** — shape, columns, summary stats, missing values, duplicates
2. **Visualization** — count plots, histograms with KDE, pair plots colored by personality type
3. **Statistical feature selection** — one-way ANOVA (F-test) on every feature grouped by personality type
4. **Encoding**
   - Target (`personality_type`) encoded with `LabelEncoder`
   - Numeric features scaled with `StandardScaler` (mean = 0, std = 1)
5. **Feature selection** — dropped statistically insignificant features: `emotional_stability`, `stress_handling`, `creativity`
6. **Train/test split** — 80% training / 20% testing (`random_state=42`)
7. **Modeling** — Logistic Regression (multi-class)
8. **Evaluation** — Accuracy score + Confusion Matrix
9. **Persistence** — trained `model.pkl` and `scaler.pkl` saved with pickle

---

## 📊 Results

### One-way ANOVA — Top Predictive Features (F-statistic, p ≤ 0.05)

| Feature | F-statistic | p-value | Significant |
| :--- | ---: | ---: | :---: |
| `party_liking` | 28878.23 | ≈ 0.0 | ✅ |
| `alone_time_preference` | 19614.80 | ≈ 0.0 | ✅ |
| `public_speaking_comfort` | 19578.06 | ≈ 0.0 | ✅ |
| `talkativeness` | 19446.84 | ≈ 0.0 | ✅ |
| `excitement_seeking` | 19160.46 | ≈ 0.0 | ✅ |
| `social_energy` | 18674.73 | ≈ 0.0 | ✅ |
| `creativity` | 0.0306 | 0.970 | ❌ dropped |
| `stress_handling` | 0.7952 | 0.451 | ❌ dropped |
| `emotional_stability` | 1.8145 | 0.163 | ❌ dropped |

### Model Performance (Logistic Regression)

| Metric | Score |
| :--- | ---: |
| **Test Accuracy** | **0.9975** |
| Train samples | 16,000 |
| Test samples | 4,000 |
| Classes | 3 |

### Confusion Matrix (test set)

```
                 Predicted
              Intro   Ambi   Extro
Actual Intro  [ ~1330    ~5     0  ]
       Ambi   [   ~1   ~1330   ~3  ]
       Extro  [    0     ~1   ~1330 ]
```

> **Best model:** Logistic Regression with `StandardScaler` on 26 selected features — **99.75% test accuracy**, with almost no misclassifications.

---

## 💡 Key Learnings

- **ANOVA is a great first filter** — it told us upfront which of the 29 features actually separate the three personality classes
- **Social/behavioral traits dominate** — `party_liking`, `talkativeness`, `social_energy`, and `alone_time_preference` are by far the most predictive
- **Not every feature helps** — `creativity`, `emotional_stability`, and `stress_handling` had p > 0.05, so dropping them simplified the model without losing accuracy
- **Scaling matters** — `StandardScaler` keeps Logistic Regression stable and speeds up convergence
- **Logistic Regression can still win** — on linearly separable synthetic data, it performed exceptionally well
- **Saving artifacts** (`model.pkl`, `scaler.pkl`) makes the model reusable for inference without re-training

---

## ⚠️ Limitations & Future Work

- Dataset is **synthetic** — real-world accuracy may be lower; validate on real survey data
- Features are already numeric — real data would need more cleaning and text/NLP handling
- Model reaches near-perfect accuracy — **k-fold cross-validation** would give a more robust estimate
- Try comparing with **Random Forest, XGBoost, and SVM** to confirm Logistic Regression is truly best
- Add **precision, recall, and F1-score** for each personality class (via `classification_report`)
- Deploy as a **Streamlit / Flask** web app for interactive predictions
- Add **feature importance analysis** using Logistic Regression coefficients

---

## 📁 Project Structure

```
personality-predictor/
├── data/
│   └── personality_synthetic_dataset.csv
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── logisticRegressionModel.ipynb
├── src/
│   ├── __init__.py
│   └── predictor.py
├── tests/
│   └── test_predictor.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── app.py
```

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `SciPy` · `Jupyter Notebook`

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/personality-predictor.git
cd personality-predictor
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### 3. Open the notebook

```bash
jupyter notebook logisticRegressionModel.ipynb
```

> Make sure `personality_synthetic_dataset.csv` is in the same folder as the notebook. The notebook also saves `model.pkl` and `scaler.pkl` after training.

### 4. Use the saved model directly

```python
import joblib
import numpy as np

model  = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# 29 features in the same order as training
sample = np.array([[6.7, 3.8, 8.7, 2.5, 7.0, 8.5, 6.7, 6.4, 6.1,
                    0.0, 8.4, 6.8, 8.6, 5.5, 4.2, 8.9, 8.3, 4.9,
                    4.4, 4.8, 8.2, 5.2, 10.0, 9.1, 4.8, 9.1,
                    8.3, 8.0, 7.1]])

pred = model.predict(scaler.transform(sample))
print(pred)
```

---

## 📬 Connect

Built while learning machine learning. Feedback and suggestions are welcome!

- **LinkedIn:** [Your Name](https://linkedin.com/in/your-linkedin)
- **GitHub:** [YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.