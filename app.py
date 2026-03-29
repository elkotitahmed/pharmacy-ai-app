import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# تحميل الداتا
df = pd.read_csv('drug200.csv')

# Encoding
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['BP'] = le_bp.fit_transform(df['BP'])
df['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])
df['Drug'] = le_drug.fit_transform(df['Drug'])

# تدريب الموديل
X = df.drop('Drug', axis=1)
y = df['Drug']

model = DecisionTreeClassifier()
model.fit(X, y)

# واجهة التطبيق
st.title("💊 Smart Pharmacy Assistant")

age = st.slider("Age", 1, 100, 30)
sex = st.selectbox("Sex", ["F", "M"])
bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.slider("Na_to_K Ratio", 5.0, 30.0, 10.0)

# تحويل المدخلات
sex = le_sex.transform([sex])[0]
bp = le_bp.transform([bp])[0]
chol = le_chol.transform([chol])[0]

# توقع
if st.button("Predict Drug"):
    input_data = pd.DataFrame([[age, sex, bp, chol, na_to_k]],
                              columns=X.columns)

    prediction = model.predict(input_data)
    result = le_drug.inverse_transform(prediction)

    st.success(f"💊 Recommended Drug: {result[0]}")
