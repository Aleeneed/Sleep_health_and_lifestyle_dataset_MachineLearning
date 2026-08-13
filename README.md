#  Sleep Health and Lifestyle Machine Learning Analysis

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 專案簡介 (Project Overview)

本專案旨在透過**機器學習 (Machine Learning)** 演算法，深入分析人類的**睡眠品質、生活習慣與生理指標**之間的關聯性，並建立預測模型以判定個體是否患有**睡眠障礙 (Sleep Disorder)**，如失眠 (Insomnia) 或睡眠呼吸中止症 (Sleep Apnea)。

隨著現代生活節奏加快與工作壓力增加，睡眠問題已成為全人類面臨的重要健康議題。本專案透過資料探勘與機器學習，企圖解析影響睡眠品質的核心因子（如壓力指數、運動量、職業別、血壓與心率等），並提供數據驅動的健康改善建議。

---

##  資料集說明 (Dataset Overview)

本專案採用 Kaggle 經典的 **Sleep Health and Lifestyle Dataset**，包含 **374 個樣本資料**與 **13 個欄位特徵**。

### 特徵欄位與說明 (Feature Descriptions)

| 欄位名稱 (Column Name) | 資料型態 | 描述 (Description) |
| :--- | :--- | :--- |
| `Person ID` | 數值 (int) | 個體唯一標識符 (Unique Identifier) |
| `Gender` | 類別 (string) | 性別 (`Male` / `Female`) |
| `Age` | 數值 (int) | 年齡 (歲) |
| `Occupation` | 類別 (string) | 職業類型 (如 Engineer, Doctor, Nurse, Teacher 等 11 種職業) |
| `Sleep Duration` | 數值 (float) | 每日平均睡眠時間 (小時) |
| `Quality of Sleep` | 數值 (int) | 睡眠品質自我評分 (1 - 10 分) |
| `Physical Activity Level` | 數值 (int) | 每日物理運動時間 (分鐘/天) |
| `Stress Level` | 數值 (int) | 自我感知壓力等級 (1 - 10 分) |
| `BMI Category` | 類別 (string) | 體重指數類別 (`Normal`, `Normal Weight`, `Overweight`, `Obese`) |
| `Blood Pressure` | 類別 (string) | 血壓測量值 (格式為 `收縮壓/舒張壓`，如 `126/83`) |
| `Heart Rate` | 數值 (int) | 安靜心率 (bpm) |
| `Daily Steps` | 數值 (int) | 每日步行步數 |
| **`Sleep Disorder`** | **目標變數** | **睡眠障礙狀況 (`None`, `Insomnia`, `Sleep Apnea`)** |

---

## 資料預處理與特徵工程 (Data Preprocessing & Feature Engineering)

為了提升機器學習模型的表現與穩定度，專案中執行了以下預處理步驟：

1. **缺失值處理 (Missing Values)**：
   - `Sleep Disorder` 欄位中的缺失值 (NaN) 表示該受試者「無睡眠障礙」，統一填補為 `None`。
2. **文本正規化與標籤統一 (Data Cleaning)**：
   - 將 `BMI Category` 中的 `Normal Weight` 與 `Normal` 進行統一合併為 `Normal`。
3. **血壓欄位特徵拆分 (Feature Extraction)**：
   - 將原始格式為 `120/80` 的 `Blood Pressure` 拆解為兩個獨立的數值型特徵：**收縮壓 (`Systolic_BP`)** 與 **舒張壓 (`Diastolic_BP`)**。
4. **類別特徵編碼 (Categorical Encoding)**：
   - 使用 **One-Hot Encoding** 或 **Label Encoding** 處理 `Gender`, `Occupation`, `BMI Category` 等類別變數。
5. **資料特徵標準化 (Feature Scaling)**：
   - 使用 `StandardScaler` / `MinMaxScaler` 對連續型變數（如步數、心率、睡眠時間等）進行縮放，避免特徵尺度影響模型距離計算。

---

## 機器學習模型與工作流程 (ML Pipeline)

本專案建立了一個完整的機器學習分類工作流程：

Sleep_health_and_lifestyle_dataset_MachineLearning/
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv  # 原始資料集
├── notebooks/
│   └── sleep_health_analysis.ipynb             # EDA 與機器學習模型訓練 Notebook
├── src/
│   ├── preprocessing.py                        # 資料預處理腳本
│   └── train.py                                # 模型訓練與評估腳本
├── README.md                                   # 專案說明文件
└── requirements.txt                            # 環境套件依賴清單

