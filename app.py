import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="RiskVision",
    layout="wide"
)

# ------------------- PATHS -------------------
MODEL_PATH = "models/"
DATA_PATH = "data/creditcard_sample.csv"

# ------------------- SAFETY CHECKS -------------------
assert os.path.exists(MODEL_PATH + "fraud_xgb_model.pkl"), "Model file missing"
assert os.path.exists(MODEL_PATH + "feature_selector.pkl"), "Feature selector missing"
assert os.path.exists(MODEL_PATH + "scaler.pkl"), "Scaler missing"
assert os.path.exists(DATA_PATH), "Dataset missing"

# ------------------- LOAD ARTIFACTS -------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH + "fraud_xgb_model.pkl")
    selector = joblib.load(MODEL_PATH + "feature_selector.pkl")
    scaler = joblib.load(MODEL_PATH + "scaler.pkl")
    return model, selector, scaler

model, selector, scaler = load_artifacts()

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

X_raw, y = load_data()

# ------------------- PREPROCESS (MATCH TRAINING) -------------------
X_selected = selector.transform(X_raw.values)
selected_features = X_raw.columns[selector.support_]

X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

scale_col = scaler.feature_names_in_[0]  # Amount
X_selected_df[scale_col] = scaler.transform(
    X_selected_df[[scale_col]]
)

X_final = X_selected_df.values

# ------------------- FIXED THRESHOLD -------------------
THRESHOLD = 0.50

# ------------------- PREDICTIONS -------------------
y_proba = model.predict_proba(X_final)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

# ------------------- TITLE -------------------
st.title("***RiskVision***")
st.subheader("A Real-Time Fraud Risk Dashboard")


st.markdown(
    """
    This dashboard enables **real-time fraud risk analysis**
    with transaction-level prediction and performance insights.
    """
)

# ------------------- KPI METRICS -------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Precision", f"{precision_score(y, y_pred):.2f}")
c2.metric("Recall", f"{recall_score(y, y_pred):.2f}")
c3.metric("F1 Score", f"{f1_score(y, y_pred):.2f}")
c4.metric("ROC-AUC", f"{roc_auc_score(y, y_proba):.2f}")

st.divider()

# ------------------- INTERACTIVE CONFUSION MATRIX -------------------
st.subheader("Interactive Confusion Matrix")

cm = confusion_matrix(y, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual: Genuine", "Actual: Fraud"],
    columns=["Predicted: Genuine", "Predicted: Fraud"]
)

fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(x="Predicted", y="Actual", color="Count")
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------- FEATURE IMPORTANCE -------------------
st.subheader("Top Feature Importances")

importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.head(10).set_index("Feature"))

# ------------------- TOP FEATURES + AMOUNT -------------------
top_features = importance_df.head(5)["Feature"].tolist()

if scale_col not in top_features:
    top_features.append(scale_col)

top_features = list(dict.fromkeys(top_features))

st.divider()

# ------------------- USER INPUT TRANSACTION SIMULATOR -------------------
st.subheader("Manual Transaction Fraud Prediction")

st.markdown(
    """
    Enter values for the **most influential features**.  
    Remaining features are auto-filled using dataset medians.
    """
)

ordered_features = [scale_col] + [f for f in top_features if f != scale_col]

user_input = {}
left, right = st.columns(2)

for i, feature in enumerate(ordered_features):
    col = left if i % 2 == 0 else right
    with col:
        if feature == scale_col:
            user_input[feature] = st.slider(
                "💰 Transaction Amount",
                min_value=float(X_raw[feature].min()),
                max_value=float(X_raw[feature].quantile(0.99)),
                value=float(X_raw[feature].median()),
                step=1.0
            )
        else:
            user_input[feature] = st.number_input(
                f"{feature} (PCA)",
                value=float(X_raw[feature].median()),
                step=0.01
            )

# ------------------- AUTO-FILL REMAINING FEATURES -------------------
full_input = {
    col: user_input.get(col, float(X_raw[col].median()))
    for col in X_raw.columns
}

user_df = pd.DataFrame([full_input])

# ------------------- SAME PIPELINE AS TRAINING -------------------
user_selected = selector.transform(user_df.values)
user_df_selected = pd.DataFrame(user_selected, columns=selected_features)

user_df_selected[scale_col] = scaler.transform(
    user_df_selected[[scale_col]]
)

user_final = user_df_selected.values

# ------------------- PREDICTION -------------------
if st.button("Predict Fraud Risk"):
    prob = model.predict_proba(user_final)[0][1]

    st.metric("Fraud Probability", f"{prob:.2%}")

    if prob >= THRESHOLD:
        st.error("High Fraud Risk Detected")
    else:
        st.success("Transaction Appears Legitimate")

# ------------------- FOOTER -------------------
st.markdown(
    """
    ---
    **XGBoost + RFECV + Streamlit Dashboard**  
    Developed by **Naga Niroop**
    """
)
