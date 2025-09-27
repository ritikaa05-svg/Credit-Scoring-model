import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\Ritika Kunwar\PycharmProjects\credit_scoring\b.csv")
print(df.head())
print(df.info())
print(df.describe())

credit_utilization_fig = px.box(df, y='Credit Utilization Ratio',

                                title='Credit Utilization Ratio Distribution')

credit_utilization_fig.show()
features = [
    'Credit Utilization Ratio',
    'Payment History',
    'Number of Credit Accounts',
    'Loan Amount',
    'Interest Rate',
    'Loan Term'
]
X = df[features]