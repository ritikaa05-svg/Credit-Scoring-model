import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import warnings

plt.switch_backend('Agg')

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv(r"C:\Users\Ritika Kunwar\PycharmProjects\credit_scoring\b.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: credit_scoring.csv not found.")
    print("Please update the file path in the 'pd.read_csv' line.")
    exit()

# Display the first few rows to verify loading
print("\n--- 1. Data Head ---")
print(df.head())
print("\n--- 1. Data Info ---")
df.info()

features = [
    'Credit Utilization Ratio',
    'Payment History',
    'Number of Credit Accounts',
    'Loan Amount',
    'Interest Rate',
    'Loan Term'
]
X = df[features]

print(f"\n--- 2. Features Selected for Clustering: {features} ---")
print(X.describe().T)

# Histogram for Credit Utilization Ratio
print("\n--- 2.1. Visualization: Credit Utilization Ratio Distribution ---")
# Using Plotly for interactive visualization (should work without GUI backend)
fig_hist = px.histogram(
    df,
    x='Credit Utilization Ratio',
    title='Distribution of Credit Utilization Ratio',
    nbins=30,
    template='plotly_white'
)


# Correlation Heatmap
print("\n--- 2.2. Visualization: Feature Correlation Heatmap ---")
plt.figure(figsize=(10, 8))
correlation_matrix = X.corr()
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=.5
)
plt.title('Correlation Heatmap of Selected Features')
# Saving the figure instead of showing it to avoid TclError
plt.savefig('correlation_heatmap.png')
print("Saved Correlation Heatmap to correlation_heatmap.png")
plt.close() # Close the figure to free up memory
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

print("\n--- 3. Data Scaling Complete ---")
print(f"Shape of scaled features: {scaled_features.shape}")
wcss = []
for k in range(1, 11):
    # n_init=10 is standard practice
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

print("\n--- 4.1. Visualization: Elbow Method for Optimal K ---")
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='darkcyan')
plt.title('Elbow Method to Determine Optimal Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.savefig('elbow_method_plot.png')
print("Saved Elbow Method Plot to elbow_method_plot.png")
plt.close()

# we often choose k=4 to represent High-Risk, Medium-Risk, Low-Risk, and Excellent segments.
optimal_k = 4
print(f"\n--- 4.2. Optimal K selected: {optimal_k} ---")

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Credit Score Segment'] = final_kmeans.fit_predict(scaled_features)

print("\n--- 5. Segment Assignment Complete ---")
print(df['Credit Score Segment'].value_counts())

segment_profiles = df.groupby('Credit Score Segment')[features].mean()
print("\n--- 6. Segment Profiles (Mean Feature Values) ---")
print("Analyze these values to label the segments (e.g., High Risk, Low Risk):")
print(segment_profiles)
print("\n--- 7. Visualization: 3D Scatter Plot of Final Segments ---")
fig_3d = px.scatter_3d(
    df,
    x='Credit Utilization Ratio',
    y='Payment History',
    z='Interest Rate',
    color='Credit Score Segment',
    title='3D Visualization of Credit Score Segments',
    color_continuous_scale=px.colors.qualitative.Bold,
    opacity=0.8,
    template='plotly_dark'
)
# fig_3d.show() # Uncomment this if you run in a jupyter notebook or environment that supports Plotly output
print("Plotly 3D scatter plot generated. Run in an environment supporting Plotly or save to HTML.")


import joblib
joblib.dump(final_kmeans, 'kmeans_credit_model.joblib')
joblib.dump(scaler, 'scaler_credit_data.joblib')
print("\n--- 8. Deployment Assets Saved ---")
print("kmeans_credit_model.joblib and scaler_credit_data.joblib saved successfully.")
