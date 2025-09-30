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

### Performance Comparison - All Models

| Model | Training ROC AUC | Test ROC AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|-------|------------------|--------------|---------------|----------------|-------------|---------------|
| **XGBoost (Optimized)** ⭐ | **1.0000** | **0.8882** | **95.09%** | **95.54%** | **78.53%** | **94.90%** |
| XGBoost (Default) | 0.9986 | 0.8490 | 93.56% | 95.07% | 70.68% | 93.18% |
| Random Forest (Optimized) | 1.0000 | 0.8319 | 93.05% | 96.24% | 67.02% | 92.55% |
| Decision Tree | 1.0000 | 0.8626 | 91.92% | 80.77% | 76.96% | 91.85% |
| Random Forest (Default) | 1.0000 | 0.8260 | 92.74% | 95.45% | 65.97% | 92.21% |
| Gradient Boost | 0.7429 | 0.6824 | 85.89% | 77.32% | 39.27% | 83.98% |
| AdaBoost | 0.6670 | 0.6400 | 83.54% | 66.30% | 31.94% | 81.15% |
| Logistic Regression | 0.6366 | 0.6301 | 83.54% | 68.29% | 29.32% | 80.78% |

### 🏆 Best Model: XGBoost Classifier (Optimized)
**Hyperparameter Tuning Results:**
- n_estimators: 200
- max_depth: 12
- learning_rate: 0.1
- colsample_bytree: 1

### **🎯 Final ROC AUC Score: 0.8882**

**Key Performance Highlights:**
- **Excellent Discrimination**: ROC AUC of 0.8882 indicates strong ability to distinguish between customers who will/won't purchase
- **High Precision**: 95.54% precision minimizes false positives (wasted marketing spend)
- **Balanced Performance**: Good balance between precision and recall for practical business application
- **Significant Improvement**: 40.8% improvement in ROC AUC over baseline Logistic Regression

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