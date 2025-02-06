# Quantitative Research Challenge: ML-Based Momentum Trading Strategy

## 📌 Overview  
This project explores a **Momentum-Based Trading Strategy** using **Machine Learning**, implementing a **Voting Classifier** to predict market movements. By leveraging **historical price data**, we extract meaningful features to make informed trading decisions, aiming to outperform traditional momentum strategies.  

## 📊 Dataset  
- **Source**: Downloaded from [Yahoo Finance](https://finance.yahoo.com/) using `yfinance`.  
- **Training Data**: 1st Jan 2015 - 1st Jan 2025  
- **Test Data**: 1st Jan 2025 - 30th Jan 2025  
- **Raw Features**: `Open`, `Close`, `High`, `Low`, `Volume`  
- **Engineered Features**: Created using the `add_features` function, enhancing predictive power with:  
  - **Technical Indicators** (e.g., Moving Averages, RSI, MACD)  
  - **Volatility Metrics** (e.g., Bollinger Bands, ATR)  
  - **Momentum Indicators** (e.g., Rate of Change, Stochastic Oscillator)  

## 🔥 Strategy & Implementation  

### 1️⃣ Feature Engineering  
To extract market insights, we apply various feature transformations:  
- **Lagged Returns & Moving Averages**  
- **Relative Strength Index (RSI) & MACD**  
- **Bollinger Bands & Momentum Oscillators**  

### 2️⃣ Model Selection: Voting Classifier  
A **Voting Classifier** aggregates predictions from multiple ML models to enhance robustness. Our ensemble includes:  
- **Logistic Regression** – Captures linear relationships  
- **Kernel SVM** – Handles non-linearity with different kernel functions  
- **Random Forest** – Handles non-linearity and feature interactions  
- **XGBoost** – Provides strong predictive performance using gradient boosting  

For **Logistic Regression** and **Kernel SVM**, the problem is framed as a **One-vs-Rest** classification task, where a separate model is trained for each class.  

### 3️⃣ Hyperparameter Optimization & Overfitting Prevention  
- **All model parameters are optimized using Grid Search.**  
- **Cross-validation** is performed to prevent overfitting and improve generalization.  

### 4️⃣ Training & Evaluation  
- **Training Phase**: The model learns market patterns from historical data.  
- **Testing Phase**: Performance is evaluated on unseen data (Jan 2025).  
- **Metrics Used**:  
  - Accuracy  
  - Precision, Recall, F1-Score  
  - Sharpe Ratio (to assess risk-adjusted returns)  

## 🚀 Results & Insights  
- The **Voting Classifier** stabilized individual models' performance by combining their strengths.  
- **Momentum signals** from engineered features significantly improved predictive accuracy.  
- **Risk management** strategies were implemented to optimize trading decisions.  

## 🛠 Future Enhancements  
- Incorporate **Deep Learning** models (e.g., LSTMs) for sequential pattern recognition.  
- Explore **Reinforcement Learning** for dynamic portfolio optimization.  
- Introduce **Sentiment Analysis** using financial news and social media.  

## 📜 Conclusion  
This project demonstrates the potential of ML in quantitative finance, enhancing traditional momentum strategies with **data-driven insights**. By combining diverse ML models through a **Voting Classifier**, we improve predictive accuracy, making it a powerful tool for systematic trading.  

## 📧 Contact

If you have questions, suggestions, or just want to connect, feel free to reach out!

- **Name**: Manoj Kumar.CM  
- **Email**: [manoj.kumar@dsai.iitm.ac.in]  
- **GitHub Profile**: [Manoj Kumar C M](https://github.com/MANOJKUMAR-CM)
---  
💡 **Have feedback or suggestions? Feel free to contribute!**  

