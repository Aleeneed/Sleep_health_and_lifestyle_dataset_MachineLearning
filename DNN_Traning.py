import pandas as pd
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   



df = pd.read_csv("D:/python/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv")
df = df.drop(["Person ID", "Occupation", "Blood Pressure", "Heart Rate","Sleep Disorder"], axis=1)
le = LabelEncoder()
enc_col = ["Gender","BMI Category"]
for col in enc_col:
    df[col] = le.fit_transform(df[col])
# --- Feature and Label Splitting ---
features = df.drop(["Stress Level"], axis=1)
labels = df["Stress Level"] #應變數
print(features.shape)
print(labels.shape)
# '隨機抽樣'
trainX, testX, trainY, testY = train_test_split(features, labels, test_size = 0.3, random_state = 42)
trainY = np.array(trainY).reshape(-1, 1)
testY = np.array(testY).reshape(-1, 1)

#建模
model = Sequential()
model.add(Dense(128, input_dim=features.shape[1], activation='relu'))  # 自動抓輸入維度
model.add(Dropout(0.5))
model.add(Dense(1)) 
model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.01),
              metrics=['mse','mape'])

dnn = model.fit(trainX, trainY, epochs=40, batch_size=30)
predictions = model.predict(testX)
# print(predictions)

# 計算評估指標
mae = mean_absolute_error(testY, predictions)
mse = mean_squared_error(testY, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(testY, predictions)

# 印出結果
print("\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R^2): {r2:.4f}")
print("--------------------------------")
