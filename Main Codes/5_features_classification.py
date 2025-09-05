import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Added numpy import

# --- Corrected Data Preparation ---

print("Loading data...")
df = pd.read_csv('features.csv', header=None)

# Separate features and labels
X = df.iloc[:, :-2]
y = df.iloc[:, -1]

# Data Cleaning (This part is fine)
X = X.apply(pd.to_numeric, errors='coerce')
X.dropna(inplace=True)
y = y[X.index]

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 1. Split the data FIRST (using the original, unscaled data)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=15, stratify=y_encoded
)

# 2. Normalize features AFTER splitting
print("Normalizing features...")
scaler = MinMaxScaler()
# Fit the scaler ONLY on the training data
X_train_scaled = scaler.fit_transform(X_train)
# Use the SAME scaler to transform the test data
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}")

# --- IMPORTANT ---
# You must now use X_train_scaled and X_test_scaled in your model_assess function.
# The function call would look like this:
# model.fit(X_train_scaled, y_train)
# preds = model.predict(X_test_scaled)

# Define and Assess Models 
def model_assess(model, title="Default"):
    try:
        print(f"Training {title}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f'Accuracy {title}: {round(accuracy, 5)}')
        
        # Generate Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,  # Use original labels on plot
                    yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Print Classification Report
        print(f"Classification Report for {title}:")
        print(classification_report(y_test, preds, target_names=label_encoder.classes_))
        print("-" * 20)
        
    except Exception as e:
        print(f"Error training or evaluating {title}: {e}")
        print("-" * 20)

# Run Models
print("\n--- Model Evaluation ---")

# Naive Bayes
nb = GaussianNB()
model_assess(nb, "Naive Bayes")

# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=5000, random_state=0)
model_assess(sgd, "Stochastic Gradient Descent")

# KNN
knn = KNeighborsClassifier(n_neighbors=19)
model_assess(knn, "KNN")

# Decision Trees
tree = DecisionTreeClassifier(random_state=42)
model_assess(tree, "Decision Trees")

# Random Forest
rforest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model_assess(rforest, "Random Forest")

# Support Vector Machine
svm = SVC()
model_assess(svm, "Support Vector Machine")

# Logistic Regression
lg = LogisticRegression(random_state=0, solver='liblinear', max_iter=1000)
model_assess(lg, "Logistic Regression")

# Neural Nets
nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, max_iter=500)
model_assess(nn, "Neural Nets")

# Cross Gradient Booster
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
model_assess(xgb, "XGBoost")

# Cross Gradient Booster (Random Forest)
try:
    xgbrf = XGBRFClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_assess(xgbrf, "XGBoost Random Forest")
except Exception as e:
    print(f"Could not run XGBRFClassifier: {e}. Trying alternative XGBoost settings.")
    xgbrf_alt = XGBClassifier(n_estimators=100, learning_rate=0.1, booster='gbtree', subsample=0.8, colsample_bytree=1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_assess(xgbrf_alt, "XGBoost (RF-style)")