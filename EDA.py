
import pandas as pd

# Load dataset
df = pd.read_csv("/content/mobile_sales_data.csv")

# Function & control structures for price categorization
def label_price(price):
    if price > 40000:
        return 'High'
    elif price > 20000:
        return 'Medium'
    else:
        return 'Low'

df["Price_Category"] = df["Price"].apply(label_price)


import numpy as np

# Convert "Inward Date" and "Dispatch Date" to datetime format
df["Inward Date"] = pd.to_datetime(df["Inward Date"], errors='coerce')
df["Dispatch Date"] = pd.to_datetime(df["Dispatch Date"], errors='coerce')
df["Days to Dispatch"] = (df["Dispatch Date"] - df["Inward Date"]).dt.days

# Convert "RAM" from string to numeric (float)
df["RAM"] = df["RAM"].str.replace("GB", "", regex=False)
df["RAM"] = pd.to_numeric(df["RAM"], errors='coerce')  # Added error handling for RAM conversion

# Function to convert storage to GB
def convert_storage(val):
    if pd.isna(val): return np.nan
    val = val.upper()
    if "TB" in val:
        return float(val.replace("TB", "")) * 1024
    elif "GB" in val:
        return float(val.replace("GB", ""))
    return np.nan

# Apply ROM conversion
df["ROM"] = df["ROM"].apply(convert_storage)

# Drop rows with missing values in essential columns
df = df.dropna(subset=["Price", "Quantity Sold", "RAM", "ROM", "Days to Dispatch"])


import matplotlib.pyplot as plt

# 1. Line Plot
df_sorted = df.sort_values(by="Inward Date")
plt.figure(figsize=(10,5))
plt.plot(df_sorted["Inward Date"], df_sorted["Quantity Sold"])
plt.title("Quantity Sold Over Time")
plt.xlabel("Date")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import seaborn as sns

# 2. Bar Plot
plt.figure(figsize=(8,5))
region_sales = df.groupby("Region")["Quantity Sold"].sum().reset_index()
sns.barplot(data=region_sales, x="Region", y="Quantity Sold")
plt.title("Total Sales by Region")
plt.show()

# 3. Histogram
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("Distribution of Mobile Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


# 4. Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Quantity Sold"])
plt.title("Outliers in Quantity Sold")
plt.show()

# 5. Violin Plot
plt.figure(figsize=(8,5))
sns.violinplot(x="Price_Category", y="Quantity Sold", data=df)
plt.title("Quantity Sold by Price Category")
plt.show()


# 6. Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x="Price", y="Quantity Sold", hue="Region", data=df)
plt.title("Price vs Quantity Sold")
plt.show()

# 7. Pair Plot
sns.pairplot(df[["Price", "RAM", "ROM", "Quantity Sold"]])
plt.suptitle("Pair Plot of Key Features", y=1.02)
plt.show()


# 8. Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[["Price", "Quantity Sold", "RAM", "ROM", "Days to Dispatch"]].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 9. Pie Chart
region_counts = df["Region"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Product Entry Distribution by Region")
plt.axis('equal')
plt.show()


from sklearn.decomposition import PCA  # Import PCA here

# Select numerical features for PCA
numerical_features = ["Price", "RAM", "ROM", "Days to Dispatch"]
X_pca = df[numerical_features]

# Apply PCA for 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_pca)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

# Add Quantity Sold for visualization
principalDf['Quantity Sold'] = df['Quantity Sold']

# Scatter plot of PCA projection
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Quantity Sold', data=principalDf, palette='viridis')
plt.title('PCA Projection of Mobile Sales Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# Summary Statistics
print("Summary Statistics:\n", df.describe())

# Correlation Matrix
print("Correlation Matrix:\n", df[["Price", "RAM", "ROM", "Quantity Sold", "Days to Dispatch"]].corr())

# Covariance between Price and Quantity Sold
print("Covariance (Price & Quantity Sold):", df["Price"].cov(df["Quantity Sold"]))


import scipy.stats as stats

# Shapiro-Wilk Test for normality
stat, p = stats.shapiro(df["Price"])
print(f"Shapiro-Wilk Test for Price: p-value = {p:.4f}")

# T-test: High vs Low Price based on Quantity Sold
df["HighPrice"] = df["Price"] > df["Price"].median()
high = df[df["HighPrice"] == True]["Quantity Sold"]
low = df[df["HighPrice"] == False]["Quantity Sold"]
t_stat, p_val = stats.ttest_ind(high, low)
print(f"T-Test (High vs Low Price) p-value = {p_val:.4f}")

# Chi-Square Test between Price Category and Region
contingency = pd.crosstab(df["Price_Category"], df["Region"])
chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
print(f"Chi-Square Test (Price_Category vs Region): p-value = {chi_p:.4f}")

# Normal Distribution Curve
from scipy.stats import norm  # Import norm here
mu, std = df["Price"].mean(), df["Price"].std()
x = np.linspace(mu - 4*std, mu + 4*std, 100)
plt.figure(figsize=(8,5))
plt.plot(x, norm.pdf(x, mu, std))
plt.title("Normal Distribution (Mobile Prices)")
plt.xlabel("Price")
plt.ylabel("Probability Density")
plt.show()

# A/B Test for South vs North Regions
if "North" in df["Region"].unique() and "South" in df["Region"].unique():
    groupA = df[df["Region"] == "South"]["Quantity Sold"]
    groupB = df[df["Region"] == "North"]["Quantity Sold"]
    ab_t, ab_p = stats.ttest_ind(groupA.dropna(), groupB.dropna())
    print(f"A/B Test (South vs North): p-value = {ab_p:.4f}")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare the data for the model
df_model = df[["Price", "Quantity Sold", "RAM", "ROM", "Days to Dispatch", "Region"]].copy()
df_model = pd.get_dummies(df_model, columns=["Region"], drop_first=True)

X = df_model.drop("Quantity Sold", axis=1)
y = df_model["Quantity Sold"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n🔍 Linear Regression Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

