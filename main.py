import os
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# Éviter le warning oneDNN
# -----------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("Trading Bot + IA EUR/USD")
st.write("Affiche les signaux SMA et prédit la tendance du lendemain avec un modèle LSTM.")

# -----------------------------
# Paramètres
# -----------------------------
st.sidebar.header("Paramètres")
days = st.sidebar.slider("Nombre de jours à récupérer", min_value=60, max_value=730, value=365)

end_date = datetime.today()
start_date = end_date - timedelta(days=days)

# -----------------------------
# Récupération des données
# -----------------------------
data = yf.download("EURUSD=X", start=start_date, end=end_date)

if data.empty:
    st.error("Impossible de récupérer les données EUR/USD.")
    st.stop()

# Flatten columns si MultiIndex
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# Garder uniquement les colonnes nécessaires
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)

st.subheader("Données récentes")
st.dataframe(data.tail())

# -----------------------------
# Calcul des indicateurs techniques
# -----------------------------
# SMA
data['SMA_5'] = data['Close'].rolling(5).mean()
data['SMA_20'] = data['Close'].rolling(20).mean()

# RSI
delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
RS = roll_up / roll_down
data['RSI'] = 100 - (100 / (1 + RS))
data.fillna(0, inplace=True)

# -----------------------------
# Graphiques
# -----------------------------
st.subheader("Graphique EUR/USD - Clôture et SMA")
st.line_chart(data['Close'])
st.line_chart(data[['SMA_5','SMA_20']])

# -----------------------------
# Signal simple SMA
# -----------------------------
if data['SMA_5'].iloc[-1] > data['SMA_20'].iloc[-1]:
    sma_signal = "Aujourd'hui ça va monter : **BUY** 📈"
else:
    sma_signal = "Aujourd'hui ça va baisser : **SELL** 📉"

st.subheader("Signal SMA")
st.markdown(sma_signal)

# -----------------------------
# Préparer les données pour LSTM
# -----------------------------
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

sequence_length = 60
X = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
X = np.array(X)

# -----------------------------
# Création modèle LSTM
# -----------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# Entraînement et prédiction LSTM
# -----------------------------
if st.button("Entraîner le modèle LSTM"):
    y = (data['Close'].shift(-1) > data['Close']).astype(int)[sequence_length:]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"Modèle entraîné ! Accuracy test: {accuracy*100:.2f}%")
    
    last_sequence = X[-1].reshape(1, sequence_length, len(features))
    pred = model.predict(last_sequence, verbose=0)[0][0]
    
    st.subheader("Prédiction LSTM pour demain")
    if pred > 0.5:
        st.markdown("Hausse probable : **BUY** 📈")
    else:
        st.markdown("Baisse probable : **SELL** 📉")
