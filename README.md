# Heart Disease Classification Using Random Forest

A comprehensive machine learning project for predicting heart disease using clinical features, featuring extensive exploratory data analysis, hyperparameter tuning, and model evaluation with visualizations.

## Overview

This project implements a Random Forest classifier to predict heart disease presence based on 11 clinical features. The implementation includes data exploration, preprocessing, hyperparameter optimization, and comprehensive model evaluation with multiple performance metrics and visualizations.

## Dataset Features

The dataset contains the following 11 clinical features for heart disease prediction:

### 1. Age
- **Type**: Numerical (continuous)
- **Range**: Typically 28-77 years
- **Significance**: Age is a major risk factor for cardiovascular disease, with risk increasing significantly after age 45 for men and 55 for women

### 2. Sex
- **Type**: Categorical (Binary)
- **Values**: 
  - `M` = Male
  - `F` = Female
- **Significance**: Men typically have higher risk of heart disease at younger ages, though risk equalizes post-menopause for women

### 3. Chest Pain Type (ChestPainType)
- **Type**: Categorical (4 categories)
- **Values**:
  - `NAP` = Non-Anginal Pain (chest pain not related to heart)
  - `ASY` = Asymptomatic (no chest pain)
  - `TA` = Typical Angina (classic chest pain related to heart disease)
  - `ATA` = Atypical Angina (chest pain that may or may not be heart-related)
- **Significance**: Typical angina strongly suggests coronary artery disease, while asymptomatic cases can still have significant heart disease (silent ischemia)

### 4. Resting Blood Pressure (RestingBP)
- **Type**: Numerical (continuous)
- **Unit**: mmHg (millimeters of mercury)
- **Range**: Typically 94-200 mmHg
- **Significance**: Systolic blood pressure measurement when the heart is at rest. Values >140 mmHg indicate hypertension, a major cardiovascular risk factor

### 5. Serum Cholesterol
- **Type**: Numerical (continuous)
- **Unit**: mg/dl (milligrams per deciliter)
- **Range**: Typically 126-564 mg/dl
- **Significance**: Total cholesterol levels; values >200 mg/dl are considered elevated and increase cardiovascular risk

### 6. Fasting Blood Sugar (FastingBS)
- **Type**: Categorical (Binary)
- **Values**:
  - `1` = Fasting blood sugar > 120 mg/dl (indicates diabetes/prediabetes)
  - `0` = Fasting blood sugar ≤ 120 mg/dl (normal)
- **Significance**: Diabetes significantly increases cardiovascular disease risk

### 7. Resting Electrocardiographic Results (RestingECG)
- **Type**: Categorical (3 categories)
- **Values**:
  - `Normal` = Normal
  - `ST` = ST-T wave abnormality (T wave inversions and/or ST elevation/depression)
  - `LVH` = Left ventricular hypertrophy (enlarged heart muscle)
- **Significance**: ECG abnormalities can indicate existing heart damage or increased workload

### 8. Maximum Heart Rate Achieved (MaxHR)
- **Type**: Numerical (continuous)
- **Unit**: beats per minute (bpm)
- **Range**: Typically 71-202 bpm
- **Significance**: Lower maximum heart rate during exercise may indicate reduced cardiac fitness or blocked arteries

### 9. Exercise Induced Angina (ExerciseAngina)
- **Type**: Categorical (Binary)
- **Values**:
  - `Y` = Yes (chest pain during exercise)
  - `N` = No (no chest pain during exercise)
- **Significance**: Exercise-induced angina strongly suggests coronary artery disease

### 10. ST Depression (Oldpeak)
- **Type**: Numerical (continuous)
- **Unit**: millimeters (mm)
- **Range**: Typically 0-6.2 mm
- **Significance**: ST depression induced by exercise relative to rest. Greater depression indicates more severe coronary artery disease

### 11. ST Segment Slope (ST_Slope)
- **Type**: Categorical (3 categories)
- **Values**:
  - `Up` = Upsloping (normal response to exercise)
  - `Flat` = Flat (abnormal, suggests heart disease)
  - `Down` = Downsloping (strongly abnormal, high likelihood of severe heart disease)
- **Significance**: The slope of the peak exercise ST segment; flat or downsloping patterns suggest coronary artery disease

## Target Variable
- **HeartDisease**: Binary (1 = heart disease present, 0 = no heart disease)
- **Distribution**: ~44.7% positive cases in the dataset

## Data Quality Issues and Limitations

### Missing Critical Features

1. **Incomplete Blood Pressure Data**: Only systolic blood pressure (RestingBP) is provided
   - **Missing**: Diastolic blood pressure
   - **Impact**: Cannot calculate Mean Arterial Pressure (MAP) or Pulse Pressure, which are important cardiovascular indicators
   - **Suggested Feature Engineering**: MAP = (Systolic + 2×Diastolic) / 3

2. **Simplified Cholesterol Data**: Only total cholesterol is provided
   - **Missing**: LDL (Low-Density Lipoprotein), HDL (High-Density Lipoprotein), Triglycerides
   - **Impact**: Total cholesterol alone is less predictive than LDL/HDL ratios
   - **Clinical Significance**: High HDL is protective, while high LDL increases risk

3. **Binary Fasting Blood Sugar**: Oversimplified diabetes indicator
   - **Issue**: Binary threshold (>120 mg/dl) vs. actual glucose values
   - **Missing**: HbA1c levels for long-term glucose control assessment
   - **Impact**: Loses granularity in diabetes severity assessment

### Potentially Misleading Features

1. **Exercise Induced Angina**: May be biased
   - **Issue**: Patients with known heart problems might avoid exercise
   - **Impact**: Could create selection bias in the dataset
   - **Mitigation**: Should be interpreted alongside exercise capacity metrics

2. **Incomplete Exercise Testing**: Missing comprehensive stress test data
   - **Missing**: Exercise duration, METs achieved, reason for stopping exercise
   - **Impact**: MaxHR alone doesn't capture full exercise capacity

### Missing Important Clinical Features

1. **Cardiovascular Risk Factors**:
   - BMI (Body Mass Index)
   - Smoking history and pack-years
   - Family history of cardiovascular disease
   - Type and duration of diabetes

2. **Advanced Cardiac Markers**:
   - Pulse Wave Velocity (arterial stiffness measure)
   - Ankle-Brachial Index (peripheral artery disease indicator)
   - C-Reactive Protein (inflammation marker)
   - Troponin levels (cardiac damage indicator)

3. **Medications**:
   - Statins (cholesterol-lowering drugs)
   - ACE inhibitors or ARBs (blood pressure medications)
   - Beta-blockers
   - Anticoagulants

4. **Lifestyle Factors**:
   - Physical activity level
   - Dietary patterns
   - Alcohol consumption
   - Sleep quality metrics


## Exploratory Data Analysis (EDA)

The project performs comprehensive EDA with six key visualizations:

### 1. Target Variable Distribution
- **Pie chart** showing class balance
- **Result**: 55.3% no disease vs 44.7% heart disease (well-balanced dataset)

### 2. Age Distribution Analysis
- **Box plots** comparing age distribution between disease/no disease groups
- **Insight**: Patients with heart disease tend to be older on average

### 3. Chest Pain Type Analysis
- **Stacked bar chart** showing chest pain types vs heart disease occurrence
- **Key Finding**: Asymptomatic patients (ASY) show highest correlation with heart disease, indicating silent heart conditions

### 4. Feature Correlation Heatmap
- **Heatmap** displaying correlations between all numerical features
- **Purpose**: Identify multicollinearity and feature relationships
- **Key Insights**: Oldpeak and MaxHR show strongest correlations with target variable

### 5. Max Heart Rate vs Age
- **Scatter plot** of MaxHR vs Age, colored by heart disease status
- **Pattern**: Lower maximum heart rates at given ages correlate with heart disease presence

### 6. Exercise Angina Analysis
- **Bar chart** comparing exercise-induced angina with heart disease occurrence
- **Finding**: Strong association between exercise angina and heart disease diagnosis

## Model Selection: Why Random Forest?

### Advantages for Medical Data:

1. **Handle Mixed Data Types**: Effectively processes both numerical (age, cholesterol) and categorical (chest pain type, sex) features without extensive preprocessing

2. **Feature Importance**: Provides interpretable feature importance scores crucial for medical decision-making

3. **Robust to Outliers**: Medical data often contains outliers; Random Forest's ensemble approach reduces their impact

4. **Non-linear Relationships**: Captures complex interactions between clinical features 

5. **Reduced Overfitting**: Bootstrap aggregating reduces overfitting common in medical datasets with limited samples

6. **No Distributional Assumptions**: Unlike logistic regression, doesn't assume linear relationships or normal distributions


## Model Evaluation

### Comprehensive Evaluation Metrics:
1. **Accuracy**: Overall correctness (~89.1%)
2. **Precision**: Positive predictive value (~87.3%)
3. **Recall**: Sensitivity (~94.1%)
4. **F1-Score**: Harmonic mean of precision and recall (~90.6%)
5. **Matthews Correlation Coefficient**: Balanced measure accounting for class distribution (~78.1%)
6. **ROC AUC**: Area under receiver operating characteristic curve (~93.3%)

### Evaluation Visualizations:

1. **Confusion Matrix**: Shows true vs predicted classifications with 96% sensitivity for detecting heart disease

2. **ROC Curve**: AUC of 0.93 indicates excellent discriminative ability

3. **Feature Importance**: Top predictors align with clinical knowledge:
   - ST_Slope (ECG slope during exercise)
   - ChestPainType (especially asymptomatic)
   - ExerciseAngina (exercise-induced chest pain)
   - Oldpeak (ST depression)

4. **Learning Curve**: Shows model performance vs training set size, indicating good bias-variance tradeoff

5. **Prediction Probability Distribution**: Shows confidence levels in predictions, with good separation between classes

6. **Performance Metrics Comparison**: Visual comparison of all evaluation metrics

## Installation and Usage

### Requirements:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running the Analysis:
```python
from heart_disease_classifier import HeartDiseaseClassifier

# Initialize and run complete analysis
classifier = HeartDiseaseClassifier()
model, metrics = classifier.run_complete_analysis()
```

### Output Files:
The analysis generates several output files in the `results/` directory:
- `eda_analysis.png`: Exploratory data analysis visualizations
- `feature_importance.png`: Feature correlation with target variable
- `model_evaluation.png`: Comprehensive model evaluation plots
- `model_performance.csv`: Performance metrics summary
- `feature_importance.csv`: Random Forest feature importance scores
- `best_hyperparameters.csv`: Optimized model parameters
- `test_predictions.csv`: Predictions on test set with probabilities

## Model Performance Summary

- **Accuracy**: 89.1%
- **F1 Score**: 90.6%
- **ROC AUC**: 93.3%
- **Sensitivity**: 94.1% 
- **Specificity**: 82.9% 

The high sensitivity makes this model particularly suitable for medical screening, where missing a positive case (heart disease) has more serious consequences than a false positive.

## Future Improvements

1. **Feature Enhancement (Better Data)**: Incorporate missing clinical variables (BMI, family history, detailed lipid panels)
2. **Advanced Models**: Experiment with ensemble methods combining multiple algorithms
3. **Temporal Analysis**: Include longitudinal data for risk progression modeling
4. **External Validation**: Test on external datasets from different populations

## Clinical Disclaimer

This model is for research and educational purposes only. It should not be used for actual medical diagnosis without validation by healthcare professionals and regulatory approval.