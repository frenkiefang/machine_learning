import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('flights.csv')

# Drop rows with missing values
df_cleaned = df.dropna()

# Convert the "satisfaction" column to binary format
df_cleaned['satisfaction'] = df_cleaned['satisfaction'].apply(lambda x: 1 if x == "satisfied" else 0)

# Drop the "Unnamed: 0" column
df_cleaned = df_cleaned.drop(columns=["Unnamed: 0"])

# Splitting the dataset into training and testing sets
X = df_cleaned.drop(columns=["satisfaction"])
y = df_cleaned["satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train XGBoost
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False)
clf_xgb.fit(X_train, y_train)

# Predict on test set
y_pred = clf_xgb.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# draw the confusion matrix for the model
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


