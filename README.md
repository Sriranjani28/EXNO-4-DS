# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("/content/bmi (1).csv")

df.head()

<img width="303" alt="SS 01" src="https://github.com/user-attachments/assets/f487c023-94da-48d7-8636-6e6a390b3d78" />

df.dropna()

<img width="311" alt="SS 02" src="https://github.com/user-attachments/assets/33e9cad4-59e3-4f79-b035-a48eb97fcbed" />

max_vals=np.max(np.abs(df[['Height','Weight']]))

max_vals

<img width="367" alt="SS 03" src="https://github.com/user-attachments/assets/87633541-27bd-4152-87b6-c748f25b8bfd" />

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

df1=df.copy()

df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])

df1.head(10)

<img width="328" alt="SS 04" src="https://github.com/user-attachments/assets/50e2364d-29e0-4d71-a7ab-d48397a94f18" />

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df.head(10)

<img width="343" alt="SS 05" src="https://github.com/user-attachments/assets/f2803cd1-f82f-4255-9b0b-8ec94c058740" />

from sklearn.preprocessing import Normalizer

scaler=Normalizer()

df2=df.copy()

df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])

df2

<img width="368" alt="SS 06" src="https://github.com/user-attachments/assets/cdbdb691-bb44-482a-8c0c-3c011c1083ac" />

df3=pd.read_csv("/content/bmi (1).csv")

from sklearn.preprocessing import MaxAbsScaler

sc=MaxAbsScaler()

df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])

df3

<img width="367" alt="SS 07" src="https://github.com/user-attachments/assets/28ec86d5-22f6-4b68-9852-cde0634f0e37" />

df4=pd.read_csv("/content/bmi (1).csv")

from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()

df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])

df4.head()

<img width="360" alt="SS 08" src="https://github.com/user-attachments/assets/d54bd8d1-a906-49b2-9c4b-99ff2104fdce" />

# RESULT:
      Thus,Feature selection and Feature scaling has been used on the given dataset.
