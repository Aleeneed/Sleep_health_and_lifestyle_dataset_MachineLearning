import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Your Data Loading and Preprocessing Code ---
df = pd.read_csv("D:/python/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv")
df = df.drop(["Person ID", "Occupation", "Blood Pressure", "Heart Rate","Sleep Disorder"], axis=1)
le = LabelEncoder()
enc_col = ["Gender","BMI Category"]
for col in enc_col:
    df[col] = le.fit_transform(df[col])
# --- Feature and Label Splitting ---
features = df.drop(["Stress Level","Quality of Sleep"], axis=1)
labels = df["Stress Level"]
labels2=df["Quality of Sleep"]
# --- Train-Test Split ---
trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.3, random_state=42)
trainX1, testX1, trainY1, testY1 = train_test_split(features, labels2, test_size=0.3, random_state=42)
# --- Model Training ---
model = RandomForestClassifier(random_state=41)
model1 = RandomForestClassifier(random_state=41)
model.fit(trainX, trainY)
model1.fit(trainX1, trainY1)


# --- Prediction ---
predictions = model.predict(testX)
predictions1 = model1.predict(testX1)
# --- ✅ CORRECTED: Classification Evaluation ---
# Calculate accuracy
accuracy = accuracy_score(testY, predictions)
accuracy1 = accuracy_score(testY1, predictions1)
# Get a detailed report of precision, recall, f1-score for each class
class_report = classification_report(testY, predictions)
class_report1 = classification_report(testY1, predictions1)
# Create a confusion matrix to see where the model got confused
conf_matrix = confusion_matrix(testY, predictions)

# --- Print the Results ---
print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)
print("--------------------------------\n")
print("\n--- Model1 Evaluation Metrics ---")
print(f"Accuracy: {accuracy1:.4f}")
print("\nClassification Report:")
print(class_report1)
print("--------------------------------\n")


# # --- Visualize the Confusion Matrix ---
print("Confusion Matrix:")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
