import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------
# Load Dataset
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()

st.title("Heart Disease Prediction App")
st.write("Using Logistic Regression and Data Visualization.")

# --------------------------
# Show Raw Data
# --------------------------
if st.checkbox("Show raw dataset"):
    st.dataframe(df)

# --------------------------
# DATA VISUALIZATION SECTION
# --------------------------
st.header("📊 Data Visualization")

# --- Heart Disease Count Plot ---
st.subheader("Heart Disease Count")
fig, ax = plt.subplots()
sns.countplot(data=df, x="target", palette="viridis", ax=ax)
ax.set_xticklabels(["No Disease", "Disease"])
st.pyplot(fig)

# --- Feature Distribution ---
st.subheader("Feature Distribution")
feature = st.selectbox("Select a feature to visualize", df.columns)
fig, ax = plt.subplots()
sns.histplot(df[feature], kde=True, color="blue", ax=ax)
st.pyplot(fig)

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- Pairplot (Optional but Heavy) ---
if st.checkbox("Show Pairplot (Slow)"):
    st.subheader("Pairplot")
    fig = sns.pairplot(df, hue="target")
    st.pyplot(fig)

# --------------------------
# MODEL TRAINING
# --------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: **{accuracy:.2f}**")

# --------------------------
# Prediction Inputs
# --------------------------
st.subheader("Predict Heart Disease")

def user_input():
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
    restecg = st.selectbox("Rest ECG Results (0–2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (1=Normal, 2=Fixed defect, 3=Reversible)", [1, 2, 3])

    user_data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    return pd.DataFrame([user_data])

input_df = user_input()

# Prediction
prediction = model.predict(input_df)[0]
prediction_prob = model.predict_proba(input_df)[0][1]

if st.button("Predict"):
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Heart Disease Likely (Probability = {prediction_prob:.2f})")
    else:
        st.success(f"No Heart Disease Detected (Probability = {prediction_prob:.2f})")
