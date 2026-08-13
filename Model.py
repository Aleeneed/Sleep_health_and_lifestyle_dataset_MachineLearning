import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("D:/python/Sleep_health_and_lifestyle_dataset/Sleep_health_and_lifestyle_dataset.csv")
df = df.drop(["Person ID", "Occupation", "Blood Pressure", "Heart Rate","Sleep Disorder"], axis=1)
le = LabelEncoder()
enc_col = ["Gender","BMI Category"]
for col in enc_col:
    df[col] = le.fit_transform(df[col])

features = df.drop(["Stress Level","Quality of Sleep"], axis=1)
labels = df["Stress Level"]
labels2=df["Quality of Sleep"]
# print(features.head())
#測試集跟訓練集
trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.3, random_state=42)
trainX1, testX1, trainY1, testY1 = train_test_split(features, labels2, test_size=0.3, random_state=42)

#建模
model = RandomForestClassifier(random_state=41)
model1 = RandomForestClassifier(random_state=41)
model.fit(trainX, trainY)
model1.fit(trainX1, trainY1)

data=pd.read_csv("D:/python/Sleep_health_and_lifestyle_dataset/realdata1.csv")
le1 = LabelEncoder()
enc_col1 = ["Gender","BMI Category"]
for col1 in enc_col1:
    data[col1] = le.fit_transform(data[col1])
predictions = model.predict(data)
predictions_series = pd.Series(predictions, name='預測壓力指數(1~10)')
predictions1 = model1.predict(data)
predictions_series1 = pd.Series(predictions1, name='預測睡眠品質 (1~10)')
pre1=pd.read_csv("D:/python/Sleep_health_and_lifestyle_dataset/realdata1.csv")
pre1=pre1.rename(columns={"Gender":"性別","Age":"年齡","Sleep Duration":"睡眠時長(hr)","Physical Activity Level":"體力活動量","BMI Category":"BMI 類別","Daily Steps":"每日步數"})
pre1_reset = pre1.reset_index(drop=True)
pre_data=pd.concat([pre1_reset,predictions_series,predictions_series1],axis=1)
pre_data.to_csv("./predictions.csv",index=False, encoding='utf-8-sig')
print(pre_data)
