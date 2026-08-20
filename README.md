# Book Rating Prediction

Data cleaning, exploratory analysis, feature engineering, and rating prediction on a book reviews dataset using Python, pandas, scikit-learn, XGBoost, and LightGBM.

## Dataset

[Goodreads Books & Reviews](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) from Kaggle - a dataset containing information about books, authors, ratings, reviews, languages, publication years, and other book-related features.

## Workflow
### 1. Data cleaning
- Cleaned the dataset and prepared the features for machine learning
- Handled missing values in relevant columns
- Removed unnecessary features that were not useful for prediction
- Prepared the target variable (`average_rating`) for regression
 
### 2. Feature engineering
- Added `author_books_count` to represent how many books each author has in the dataset
- Created additional features from existing book information, such as title and original title length
- Grouped language values to reduce the number of categorical features
 
### 3. Encoding
- Applied one-hot encoding to categorical features such as language
- Converted non-numeric features into a format suitable for machine learning models
- Used information about authors as an additional feature to help the models capture author-specific patterns

### 4. Modeling
Trained and compared four regression models:
| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.207 | 0.307 |
| Random Forest | 0.201 | 0.346 |
| XGBoost | **0.199** | **0.355** |
| LightGBM | 0.200 | 0.354 |

**XGBoost** performed slightly better than the other models and was selected for feature importance analysis.

## Key findings
- `Authors` was by far the most important feature, with an importance of around 0.72 in the XGBoost model
- `language_code`, `author_books_count`, `series`, and ratings-related features had considerably smaller importance
- More complex models such as Random Forest, XGBoost, and LightGBM performed slightly better than Linear Regression
- The relatively low R² scores suggest that the available features do not capture many of the factors that actually determine how highly a book is rated, such as writing quality, story, characters, or reader preferences
- The results also show that author-specific information plays an important role in predicting book ratings

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (Linear Regression, Random Forest, preprocessing, metrics)
- XGBoost
- LightGBM

## Next steps
- Hyperparameter tuning for XGBoost and LightGBM
- Testing additional feature engineering approaches
- Exploring different encoding methods for categorical features
- Testing additional models and comparing their performance
- Analyzing whether removing or modifying the author feature improves generalization to authors not seen during training
