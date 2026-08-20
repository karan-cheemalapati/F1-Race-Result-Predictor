# F1 Race Result Predictor

Predicting Formula 1 podium finishes using machine learning on 70+ years of race data (1950–2024). Built with XGBoost and LightGBM on 22 engineered features including qualifying pace, circuit history, championship standing, and driver form — with SHAP explainability and an interactive Streamlit app.

**Live Demo:** [Click here](https://f1-race-result-predictor.streamlit.app/)
---

## Project Overview

| | |
|---|---|
| **Goal** | Predict whether a driver will finish on the podium (Top 3) |
| **Data** | 11 datasets · 26,000+ race entries · 1950–2024 |
| **Model** | XGBoost Classifier |
| **ROC-AUC** | 0.940 |
| **F1 Score** | 0.707 |

---

## Project Structure

```
F1-Race-Result-Predictor/
├── data/                        # Raw and processed datasets
├── src/
│   ├── data_loader.py           # Loads and merges all 11 CSVs
│   ├── features.py              # Feature engineering pipeline
│   └── train.py                 # Model training and evaluation
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_features.ipynb        # Feature engineering walkthrough
│   └── 03_train.ipynb           # Model training and comparison
├── models/                      # Saved trained models (.pkl)
├── app.py                       # Streamlit web app
└── requirements.txt
```

---

## Features Used (22 total)

**Race configuration**
- Grid position
- Qualifying gap to pole (seconds)
- Qualifying position

**Championship standing**
- Driver championship position & points
- Constructor championship position & points
- Points gap to championship leader
- Wins so far in the season

**Circuit history** (per driver per circuit)
- Win rate at circuit
- Podium rate at circuit
- Average finish position at circuit
- Number of races at circuit

**Form & momentum**
- Driver rolling avg finish (last 3 & 5 races)
- Constructor rolling avg points (last 3 & 5 races)

**Driver profile**
- Driver age at race
- Career experience (races started)
- Grid vs qualifying position delta

**Season context**
- Season progress (0–1)

---

## Model Results

| Model | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 0.9449 | 0.6819 | 0.5547 | 0.8847 |
| **XGBoost** ✓ | **0.9401** | **0.7067** | **0.6534** | **0.7695** |
| LightGBM | 0.9374 | 0.6918 | 0.6263 | 0.7726 |

**XGBoost** selected as the final model for best F1 score and precision balance.

**Top 5 features by SHAP importance:**
1. `driver_standing_pos` — championship leader most likely to podium
2. `pts_gap_to_leader` — gap to championship leader
3. `driver_standing_pts` — total points accumulated
4. `grid` — starting grid position
5. `constructor_standing_pts` — constructor strength

---

## Streamlit App

The interactive app lets you configure any race scenario and get an instant podium prediction with:
- Podium probability score
- Driver's circuit history chart
- Head-to-head driver comparison
- SHAP feature contribution chart

**To run locally:**

```bash
# Clone the repo
git clone https://github.com/karan-cheemalapati/F1-Race-Result-Predictor.git
cd F1-Race-Result-Predictor

# Create environment
conda create -n f1predictor python=3.11 -y
conda activate f1predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-red)
![SHAP](https://img.shields.io/badge/SHAP-0.51.0-green)

- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP
- **Visualization:** Plotly, Seaborn, Matplotlib
- **App:** Streamlit

---

## Data Sources

All datasets sourced from the [Ergast F1 API](http://ergast.com/mrd/) covering the 1950–2024 Formula 1 seasons.

---

## Author

**Karan Cheemalapati**  
MS Data Science — University of Colorado Boulder  
[GitHub](https://github.com/karan-cheemalapati) · [LinkedIn](https://linkedin.com/in/karan-cheemalapati)
