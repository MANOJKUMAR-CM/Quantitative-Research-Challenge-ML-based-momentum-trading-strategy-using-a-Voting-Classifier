import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
# Stocks
stocks = ['AAPL', 'META', 'TSLA', 'JPM', 'AMZN']

# Training Data
data = {}
for stock in stocks:
    df = yf.download(stock, start='2015-01-01', end='2025-01-01')
    data[stock] = df


def add_features(df):
    df['SMA_50'] = SMAIndicator(df['Close'].squeeze(), window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'].squeeze(), window=200).sma_indicator()
    df['EMA_21'] = EMAIndicator(df['Close'].squeeze(), window=21).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    df['MACD'] = MACD(df['Close'].squeeze()).macd()
    df['MACD_Signal'] = MACD(df['Close'].squeeze()).macd_signal()
    df['Stoch'] = StochasticOscillator(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()).stoch()
    df['BBW'] = BollingerBands(df['Close'].squeeze()).bollinger_wband()
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'].squeeze(), df['Volume'].squeeze()).on_balance_volume()

    # Return-based features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Weekly_Return'] = df['Close'].pct_change(5)
    df['Monthly_Return'] = df['Close'].pct_change(21)

    # **Drop NaNs after feature calculation**
    df.dropna(inplace=True)

    return df


for stock, df in data.items():
    data[stock] = add_features(df)
    
    
def define_labels(df):
    df['Signal'] = 0  # Hold by default
    
    # Buy Signals: Buy when RSI < 30 OR MACD crosses above Signal OR OBV rising
    buy_condition = (df['RSI'] < 30) | (df['MACD'] > df['MACD_Signal']) & (df['OBV'].diff() > 0)
    df.loc[buy_condition, 'Signal'] = 1  
    
    # Sell Signals: Sell when RSI > 70 OR MACD crosses below Signal OR OBV dropping
    sell_condition = (df['RSI'] > 70) | (df['MACD'] < df['MACD_Signal']) & (df['OBV'].diff() < 0)
    df.loc[sell_condition, 'Signal'] = -1  
    
    return df

# Applying labels to each stock
data = {stock: define_labels(df) for stock, df in data.items()}

# Preparing data for training
train_data = []
target_data = []
for stock, df in data.items():
    features = df[['SMA_50', 'SMA_200', 'EMA_21', 'RSI', 'MACD', 'MACD_Signal', 'Stoch', 'BBW', 'OBV', 'Daily_Return', 'Weekly_Return', 'Monthly_Return']]
    target = df['Signal']
    train_data.append(features)
    target_data.append(target)

X = pd.concat(train_data)
y = pd.concat(target_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGB Classifier

model_xgb = XGBClassifier(
    objective='multi:softmax',   #'multi:softmax' for multi-class
    n_estimators=700,            # number of trees
    max_depth=5,                 # tree depth
    learning_rate=0.1,           # step size
    subsample=0.8,               # fraction of samples per tree
    colsample_bytree=1,        # fraction of features per tree
    reg_alpha=0, 
    gamma = 0.2,# L1 regularization
    reg_lambda=1 ,# L2 regularization
    tree_method="hist",  # Fix for GPU warning
    device="cuda"        
    )


# Shift labels to start from 0
y_train_mapped = np.array(y_train) + 1  # [-1, 0, 1] → [0, 1, 2]

# Train the model with modified labels
model_xgb.fit(X_train, y_train_mapped)
y_pred_mapped = model_xgb.predict(X_test)
y_pred_xgb = y_pred_mapped - 1  # [0, 1, 2] → [-1, 0, 1]
print("XGB Classifier:",classification_report(y_test, y_pred_xgb))

# SVM
model_svm = Pipeline([
    ("scaler", StandardScaler()),  # Normalize data
    ("svm", OneVsRestClassifier(SVC(probability=True, kernel="rbf")))  # Use RBF kernel in One-vs-Rest
])

# Train the SVM model
model_svm.fit(X_train, y_train)

# Predict using the trained model
y_pred_svm = model_svm.predict(X_test)

print("SVM:", classification_report(y_test, y_pred_svm))

# Hyperparameter tuning -> Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
}
model_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
print("Random Forest:",classification_report(y_test, y_pred_rf))

# Logistic Regression
# Scaling
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2)

# Improved Logistic Regression with higher max_iter and lower tol
log_reg = LogisticRegression(solver='liblinear', max_iter=1000, penalty="l1", class_weight="balanced", tol=1e-5)

# Pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('poly', poly),
    ('classifier', log_reg)
])

# One-vs-Rest
model_log_reg = OneVsRestClassifier(pipeline)
model_log_reg.fit(X_train, y_train)

# Predictions
y_pred_lr = model_log_reg.predict(X_test)

# Evaluation
print("One-vs-Rest Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
print(accuracy_score(y_test, y_pred_lr))


# Soft Voting Classifier
probs_1 = model_log_reg.predict_proba(X_test)
probs_2 = model_rf.predict_proba(X_test)
probs_3 = model_svm.predict_proba(X_test)
probs_4 = model_xgb.predict_proba(X_test)

# Combine the probabilities into a 3D array (samples x models x classes)
all_probs = np.stack([probs_1, probs_2, probs_3, probs_4])

# Average the probabilities across models (axis=0)
avg_probs = np.max(all_probs, axis=0)

# Get the class with the highest average probability
final_predictions = np.argmax(avg_probs, axis=1)
final_predictions = final_predictions - 1  # [0, 1, 2] → [-1, 0, 1]

print(classification_report(y_test, final_predictions))
print(accuracy_score(y_test, final_predictions))