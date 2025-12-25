import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


datasets=pd.read_csv("heart_disease_uci.csv")


trestbps_median=datasets["trestbps"].median()
datasets["trestbps"].fillna(trestbps_median,inplace=True)
chol_median=datasets["chol"].median()
datasets["chol"].fillna(chol_median,inplace=True)
fbs_mod=datasets["fbs"].mode()[0]
datasets["fbs"].fillna(fbs_mod,inplace=True)
datasets.dropna(subset=["restecg"],inplace=True)
thalch_median=datasets["thalch"].median()
datasets["thalch"].fillna(thalch_median,inplace=True)
exang_mod=datasets["exang"].mode()[0]
datasets["exang"].fillna(exang_mod,inplace=True)
oldpeak_median=datasets["oldpeak"].median()
datasets["oldpeak"].fillna(oldpeak_median,inplace=True)

datasets=datasets.drop(columns=["id"])
datasets=datasets.drop(columns=["dataset"])
ca_median=datasets["ca"].median()
datasets["ca"].fillna(ca_median,inplace=True)

X = datasets[[
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalch","exang","oldpeak",
    "slope","ca","thal"
]]


categorical_cols = ["sex","cp","fbs","restecg","exang","slope","thal"]
for col in categorical_cols:
    X[col].fillna("Unknown", inplace=True)



y = (datasets["num"] > 0).astype(int)

X_encoded = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




