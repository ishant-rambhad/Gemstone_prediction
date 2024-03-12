import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset from a CSV file
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Title of the web application
st.title('Gemstone Prediction Dataset Dashboard')

# Load dataset
filename = r'cubic_zirconia.csv'  # Provide the path to your CSV file
df = load_data(filename)

# Display the dataframe
st.subheader('Gemstone Prediction Dataset')
st.write(df)

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[~((df[column] < (lower_bound)) | (df[column] > (upper_bound)))]
    return df_filtered

# Remove outliers from the 'carat' column
df_no_outliers = remove_outliers(df, 'carat')

# Scatter plot: Display the relationship between two numerical variables (carat vs. price) with outliers
st.subheader('Scatter Plot: Carat vs. Price (with outliers)')
scatter_fig_with_outliers = px.scatter(df, x='carat', y='price', title='Carat vs. Price (with outliers)')
st.plotly_chart(scatter_fig_with_outliers)

# Scatter plot: Display the relationship between two numerical variables (carat vs. price) without outliers
st.subheader('Scatter Plot: Carat vs. Price (without outliers)')
scatter_fig_without_outliers = px.scatter(df_no_outliers, x='carat', y='price', title='Carat vs. Price (without outliers)')
st.plotly_chart(scatter_fig_without_outliers)

# Box plot: Show the distribution of a dataset across different categories (with outliers)
st.subheader('Box Plot: Price distribution by Cut (with outliers)')
box_fig_with_outliers = px.box(df, x='cut', y='price', title='Price distribution by Cut (with outliers)')
st.plotly_chart(box_fig_with_outliers)

# Box plot: Show the distribution of a dataset across different categories (without outliers)
st.subheader('Box Plot: Price distribution by Cut (without outliers)')
box_fig_without_outliers = px.box(df_no_outliers, x='cut', y='price', title='Price distribution by Cut (without outliers)')
st.plotly_chart(box_fig_without_outliers)

# Line plot: Show trends over time or continuous variables (example: carat vs. price) with outliers
st.subheader('Line Plot: Carat vs. Price (with outliers)')
line_fig_with_outliers = px.line(df, x='carat', y='price', title='Carat vs. Price (with outliers)')
st.plotly_chart(line_fig_with_outliers)

# Line plot: Show trends over time or continuous variables (example: carat vs. price) without outliers
st.subheader('Line Plot: Carat vs. Price (without outliers)')
line_fig_without_outliers = px.line(df_no_outliers, x='carat', y='price', title='Carat vs. Price (without outliers)')
st.plotly_chart(line_fig_without_outliers)

# Bar chart: Ideal for comparing categorical data or showing counts (example: color vs. count) with outliers
st.subheader('Bar Chart: Color vs. Count (with outliers)')
bar_fig_with_outliers = px.bar(df['color'].value_counts().reset_index(), x='index', y='color', title='Color vs. Count (with outliers)')
st.plotly_chart(bar_fig_with_outliers)

# Bar chart: Ideal for comparing categorical data or showing counts (example: color vs. count) without outliers
st.subheader('Bar Chart: Color vs. Count (without outliers)')
bar_fig_without_outliers = px.bar(df_no_outliers['color'].value_counts().reset_index(), x='index', y='color', title='Color vs. Count (without outliers)')
st.plotly_chart(bar_fig_without_outliers)

# Histogram: Visualize the distribution of numerical data (example: distribution of carat) with outliers
st.subheader('Histogram: Distribution of Carat (with outliers)')
hist_fig_with_outliers = px.histogram(df, x='carat', title='Distribution of Carat (with outliers)')
st.plotly_chart(hist_fig_with_outliers)

# Histogram: Visualize the distribution of numerical data (example: distribution of carat) without outliers
st.subheader('Histogram: Distribution of Carat (without outliers)')
hist_fig_without_outliers = px.histogram(df_no_outliers, x='carat', title='Distribution of Carat (without outliers)')
st.plotly_chart(hist_fig_without_outliers)





with open("gemstones.pkl","rb") as file:
    pipeline = pickle.load(file)

def Transformation(X):
    X['carat'] = X['carat'].map({"SI1":0,"IF":1,'VVS2': 2,'VS1': 3,'VVS1': 4,'VS2': 5,'SI2': 6,'I1': 7})
    X['color'] = X['color'].map({'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6})
    X['cut'] = X['xut'].map({'Ideal': 0, 'Premium': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4})
    X = X.astype('float32')
    
    xgb_reg = pipeline.named_steps['xgb_reg']
    
    X_pca = xgb_reg.transform(X_ans_scale)
    X_ans_scale = X.drop(columns=['price'])
    X_xgb = xgb_reg.transform(X_ans_scale)

    return X_xgb


    
def GUI():

    carat = st.number_input("Enter carat (float)", min_value=0.2, step=0.01, max_value=4.5)

    cut_options = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    cut = st.selectbox("Select cut (object)", cut_options)

    color_options = ["D", "E", "F", "G", "H", "I", "J"]
    color = st.selectbox("Select color (object)", color_options)

    clarity_options = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
    clarity = st.selectbox("Select clarity (object)", clarity_options)

    depth = st.number_input("Enter depth (float)", min_value=50.8, step=0.01, max_value=73.6)
    table = st.number_input("Enter table (float)", min_value=49, max_value=79)
    x = st.number_input("Enter x (float)", min_value=0.0, step=0.01, max_value=10.23)
    y = st.number_input("Enter y (float)", min_value=0.0, step=0.01, max_value=58.9)
    z = st.number_input("Enter z (float)", min_value=0.0, step=0.01, max_value=31.8)

    if st.button("Predict price"):
        tuc = {'Ideal': 0, 'Premium': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4}
        clar = {'SI1': 0,
                'IF': 1,
                'VVS2': 2,
                'VS1': 3,
                'VVS1': 4,
                'VS2': 5,
                'SI2': 6,
                'I1': 7}
        col = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
        d1 = {
            'carat' : [carat], 
            'cut':[tuc[cut]],
            'color':[col[color]],
            'clarity':[clar[clarity]], 
            'depth':[depth], 
            'table' : [table], 
            'x':[x], 
            'y':[y], 
            'z':[z]
        }
        X = pd.DataFrame(d1)
        st.dataframe(X)
        price_prediction = pipeline.predict(X)
        st.subheader(f"Price Prediction: {price_prediction[0]:.2f}")

GUI()