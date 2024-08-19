import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Charger les données
df = pd.read_csv('Churn_Modelling.csv')


print(df.head())

# Explorer les colonnes, vérifier les valeurs manquantes, etc.
print(df.info())
print(df.describe())

features = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
target = df['Exited']

categorical_features = ['Geography', 'Gender']
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Pipeline pour prétraiter les données
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

log_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression())])

log_pipeline.fit(X_train, y_train)
log_predictions = log_pipeline.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, log_predictions))

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

rf_pipeline.fit(X_train, y_train)
rf_predictions = rf_pipeline.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, rf_predictions))

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(random_state=42))])

xgb_pipeline.fit(X_train, y_train)
xgb_predictions = xgb_pipeline.predict(X_test)
print("XGBoost:")
print(classification_report(y_test, xgb_predictions))

# Afficher les matrices de confusion
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
predictions = [log_predictions, rf_predictions, xgb_predictions]

for i, pred in enumerate(predictions):
    print(f"{models[i]} Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("accuracy_score:",accuracy_score(y_test, pred))
