# Holiday Package Purchase Prediction

## Project Overview

This project aims to help "Trips & Travel.Com" company expand their customer base by predicting which customers are most likely to purchase their new **Wellness Tourism Package**. The company previously had a low conversion rate (18%) and high marketing costs due to random customer targeting. This machine learning solution uses customer data to make marketing efforts more efficient and targeted.

## Problem Statement

"Trips & Travel.Com" currently offers 5 types of packages:
- Basic
- Standard  
- Deluxe
- Super Deluxe
- King

The company wants to launch a new **Wellness Tourism Package** and needs to identify potential customers more effectively to reduce marketing costs and improve conversion rates.

## Dataset

- **Source**: [Kaggle - Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)
- **Size**: 4,888 rows × 20 columns
- **Target Variable**: `ProdTaken` (whether customer purchased a package)

### Features Include:
- Customer demographics (Age, Gender, MaritalStatus)
- Contact information (TypeofContact, CityTier, DurationOfPitch)
- Travel preferences (PreferredPropertyStar, NumberOfTrips, Passport)
- Financial information (MonthlyIncome, OwnCar)
- Interaction data (NumberOfFollowups, PitchSatisfactionScore)

## Machine Learning Approach

### Models Evaluated:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**
5. **AdaBoost** 
6. **XGBoost** ⭐ (Best Performance)

### Data Preprocessing:
- **Categorical Features**: One-Hot Encoding (drop first)
- **Numerical Features**: Standard Scaling
- **Train-Test Split**: 80-20 ratio

### Hyperparameter Tuning:
- **Random Forest**: Tuned max_depth, max_features, min_samples_split, n_estimators
- **XGBoost**: Tuned learning_rate, max_depth, n_estimators, colsample_bytree

## Model Performance

### Best Model: XGBoost Classifier
**Optimized Parameters:**
- n_estimators: 200
- max_depth: 12
- learning_rate: 0.1
- colsample_bytree: 1

### **ROC AUC Score: 0.8882** 🎯

### Additional Metrics:
- **Accuracy**: 84.58%
- **Precision**: 69.94%
- **Recall**: 30.32%
- **F1-Score**: 82.00%

## Files Structure

```
travel_classifier/
├── Travel.csv                                          # Dataset
├── XgboostBoost Classification Implementation.ipynb    # Main analysis notebook
├── auc.png                                            # ROC curve visualization
└── README.md                                          # This file
```

## Key Insights

1. **XGBoost** achieved the highest ROC AUC score of **0.8882**, indicating excellent discrimination between customers likely to purchase vs. not purchase
2. The model can help reduce marketing costs by targeting customers with higher purchase probability
3. **Feature engineering and hyperparameter tuning** significantly improved model performance
4. The **ROC curve analysis** shows the model's strong predictive capability across different thresholds

## Business Impact

- **Targeted Marketing**: Identify high-potential customers to reduce marketing waste
- **Cost Optimization**: Focus resources on customers with higher conversion probability  
- **Revenue Growth**: Improve overall package sales through better customer targeting
- **Data-Driven Decisions**: Replace random targeting with predictive insights

## Usage

1. Load the trained XGBoost model
2. Input customer features for new prospects
3. Get probability scores for package purchase likelihood
4. Target customers with scores above optimal threshold
5. Monitor and retrain model with new data

## Technologies Used

- **Python** 🐍
- **Pandas** & **NumPy** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Matplotlib** & **Seaborn** - Data visualization
- **Plotly** - Interactive visualizations

---

**ROC AUC Score: 0.8882** - Demonstrating strong predictive performance for customer purchase behavior.