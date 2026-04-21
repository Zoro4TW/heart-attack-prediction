# Heart Attack Risk Prediction

Binary classification model to predict heart attack risk from patient health records using Random Forest with SMOTEENN balancing.

## Overview

This project develops a machine learning pipeline to predict the likelihood of heart attack events based on patient health metrics. The model addresses class imbalance through SMOTEENN (Synthetic Minority Over-sampling Technique combined with Edited Nearest Neighbors) and achieves 95.5% accuracy on test data.

## Dataset

- **Size:** 237,000 patient records (sampled to 4,000 for training)
- **Features:** Age, sex, cholesterol levels, blood pressure, smoking status, diabetes, family history, obesity, alcohol consumption, exercise habits, diet quality, stress levels, blood sugar, triglycerides, physical activity, sedentary hours, and more
- **Target:** Binary classification (heart attack risk: yes/no)

## Methodology

1. **Data Cleaning:** Handle missing values, remove duplicates, validate data types
2. **Exploratory Data Analysis (EDA):** Visualize distributions, correlations, and class balance
3. **Feature Engineering:** Create interaction features, normalize continuous variables
4. **Class Balancing:** Apply SMOTEENN to address imbalanced target variable
5. **Model Training:** Random Forest classifier with optimized hyperparameters
6. **Evaluation:** Confusion matrix, classification report, feature importance analysis

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 95.5% |
| Precision | 95.6% |
| Recall | 95.5% |
| F1-Score | 95.5% |

### Key Findings

- **Top Predictive Features:** Age, cholesterol levels, blood pressure, smoking status
- **Balanced Performance:** SMOTEENN balancing improved minority class detection without sacrificing overall accuracy
- **Generalization:** Model demonstrates strong performance on held-out test set

## Tech Stack

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning models and preprocessing
- **imbalanced-learn** - SMOTEENN implementation
- **Matplotlib/Seaborn** - Data visualization

## Project Structure

```
heart-attack-prediction/
│
├── heart_attack_prediction.ipynb    # Main notebook with full pipeline
├── patient_data.csv                  # Dataset (not included - see Data Source)
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-attack-prediction.git
   cd heart-attack-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**
   - Place your dataset as `patient_data.csv` in the project root
   - Ensure it follows the expected schema (see Dataset section)

4. **Run the notebook**
   ```bash
   jupyter notebook heart_attack_prediction.ipynb
   ```

## Dependencies

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## Future Improvements

- [ ] Experiment with ensemble methods (XGBoost, LightGBM)
- [ ] Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Feature selection using Recursive Feature Elimination (RFE)
- [ ] Cross-validation for robust performance estimation
- [ ] Deploy model as REST API using Flask/FastAPI
- [ ] Create interactive dashboard for predictions

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions or collaboration opportunities, please open an issue or reach out via [your contact method].

---

**Note:** This project is for educational and portfolio purposes. Medical predictions should always be validated by healthcare professionals.
