# 📈 Mobile Sales Forecasting using Supervised Learning and Statistical Techniques

This project focuses on analyzing and forecasting mobile phone sales using real-world-inspired data. It leverages **Python**, **pandas**, **NumPy**, **Seaborn**, **Matplotlib**, and **Scikit-learn** to perform **Exploratory Data Analysis (EDA)**, statistical testing, and predictive modeling using **Linear Regression**.

---

## 🧠 Project Objective

To analyze mobile sales data and build a supervised learning model that can forecast sales volume based on key features like **Price**, **RAM**, **ROM**, **Region**, and **Dispatch Time**.

---

## 🔍 Exploratory Data Analysis (EDA)

- Line plot of Quantity Sold over time
- Bar plot: Total sales by Region
- Pie chart: Region-wise product distribution
- Histogram: Price distribution
- Violin plot: Quantity Sold by Price Category
- Box plot: Outliers in Quantity Sold
- Scatter plot: Price vs Quantity Sold
- Pair plot of Price, RAM, ROM, Quantity Sold
- Correlation heatmap

---

## 📊 Statistical Analysis

- Shapiro-Wilk Test for normality
- T-test for High vs Low priced devices
- Chi-Square Test: Region vs Price Category
- Normal Distribution plot of Prices
- A/B Testing: South vs North region sales

---

## 🤖 Supervised Machine Learning

- Model: **Linear Regression**
- Feature Engineering and One-Hot Encoding
- Train-Test Split (80/20)
- Evaluation Metrics:
  - R² Score
  - Mean Squared Error (MSE)

---

## 🛠️ Tech Stack

- Python 3.x  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- scipy  

---

## 📁 File Structure

```bash
📦mobile-sales-forecast
 ┣ 📜 eda.py             # Contains all data preprocessing and visualizations
 ┣ 📜 model.py           # Linear Regression training and evaluation
 ┣ 📜 data.csv           # (Sample or anonymized dataset)
 ┣ 📜 README.md          # Project documentation
 ┗ 📜 requirements.txt   # Python dependencies
