import streamlit as st
import math

# ── Trained model parameters (extracted from fraud_detection_pipeline.pkl) ───

SCALER_MEAN  = [179958.3184032005, 834019.397360368,
                855217.6525841039, 1099526.04741079,
                1223809.5619918294]

SCALER_SCALE = [608213.1894787767, 2889767.4397447146,
                2925565.379696595,  3384673.7047138806,
                3656028.7694454356]

# Features: [amount_s, oldbalanceOrg_s, newbalanceOrig_s,
#            oldbalanceDest_s, newbalanceDest_s,
#            type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER]
LR_COEF      = [-0.5800585492020419,  69.96786455607014,
                -69.3179972730042,    37.513330038833,
                -40.618418082459655,  18.658903723298224,
                 -0.5850273169920219, -1.5130784716027004,
                  20.26774790657693]
LR_INTERCEPT = -20.804432564716038

# OHE: drop='first' drops CASH_IN; remaining order:
OHE_CATEGORIES = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

def predict(transaction_type, amount, oldbalanceOrg,
            newbalanceOrig, oldbalanceDest, newbalanceDest):
    # 1. Standard scale numerical features
    nums = [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    scaled = [(nums[i] - SCALER_MEAN[i]) / SCALER_SCALE[i] for i in range(5)]

    # 2. One-hot encode (drop first = drop CASH_IN)
    ohe = [1.0 if transaction_type == c else 0.0 for c in OHE_CATEGORIES]

    # 3. Dot product + intercept
    features = scaled + ohe
    logit = sum(LR_COEF[i] * features[i] for i in range(9)) + LR_INTERCEPT

    # 4. Sigmoid
    prob = 1.0 / (1.0 + math.exp(-logit))
    return (1 if prob >= 0.5 else 0), prob


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title('Fraud Detection App')
st.markdown('Enter the transaction details below and click **Predict**.')
st.divider()

transaction_type = st.selectbox('Transaction Type',
                                ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
amount         = st.number_input('Amount',                 min_value=0.0, value=1000.0)
oldbalanceOrg  = st.number_input('Old Balance (Sender)',   min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input('New Balance (Sender)',   min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input('Old Balance (Receiver)', min_value=0.0, value=0.0)
newbalanceDest = st.number_input('New Balance (Receiver)', min_value=0.0, value=0.0)

if st.button('Predict'):
    prediction, probability = predict(
        transaction_type, amount, oldbalanceOrg,
        newbalanceOrig, oldbalanceDest, newbalanceDest
    )

    st.subheader(f"Prediction : '{prediction}'")
    st.caption(f"Fraud probability: {probability:.2%}")

    if prediction == 1:
        st.error('⚠️ This transaction may be fraudulent!')
    else:
        st.success('✅ This transaction looks legitimate.')