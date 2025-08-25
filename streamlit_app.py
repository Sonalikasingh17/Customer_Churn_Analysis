import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔮 Customer Churn Prediction System")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Analytics", "About"])

if page == "Prediction":
    st.header("📈 Predict Customer Churn")
    st.write("Enter customer details to predict churn probability")

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["M", "F"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])

        st.subheader("Transaction Data")
        total_spent = st.number_input("Total Amount Spent", min_value=0.0, value=1000.0)
        avg_spent = st.number_input("Average Transaction Amount", min_value=0.0, value=100.0)
        std_spent = st.number_input("Standard Deviation of Spending", min_value=0.0, value=50.0)
        transaction_count = st.number_input("Number of Transactions", min_value=0, value=10)
        min_spent = st.number_input("Minimum Transaction Amount", min_value=0.0, value=10.0)
        max_spent = st.number_input("Maximum Transaction Amount", min_value=0.0, value=500.0)
        transaction_period_days = st.number_input("Transaction Period (Days)", min_value=0, value=365)

    with col2:
        st.subheader("Category-wise Spending")
        spent_books = st.number_input("Books Spending", min_value=0.0, value=100.0)
        spent_clothing = st.number_input("Clothing Spending", min_value=0.0, value=200.0)
        spent_electronics = st.number_input("Electronics Spending", min_value=0.0, value=300.0)
        spent_furniture = st.number_input("Furniture Spending", min_value=0.0, value=200.0)
        spent_groceries = st.number_input("Groceries Spending", min_value=0.0, value=200.0)

        st.subheader("Customer Service")
        total_interactions = st.number_input("Total Interactions", min_value=0, value=5)
        resolved_interactions = st.number_input("Resolved Interactions", min_value=0, value=4)
        resolution_rate = resolved_interactions / total_interactions if total_interactions > 0 else 0
        st.write(f"Resolution Rate: {resolution_rate:.2f}")

        st.subheader("Online Activity")
        login_frequency = st.number_input("Login Frequency", min_value=0, value=20)
        service_usage = st.selectbox("Service Usage", ["Website", "Mobile App", "Online Banking"])

    # Prediction button
    if st.button("🔮 Predict Churn"):
        try:
            # Create custom data object
            data = CustomData(
                Age=age,
                Gender=gender,
                MaritalStatus=marital_status,
                IncomeLevel=income_level,
                total_spent=total_spent,
                avg_spent=avg_spent,
                std_spent=std_spent,
                transaction_count=transaction_count,
                min_spent=min_spent,
                max_spent=max_spent,
                transaction_period_days=transaction_period_days,
                spent_books=spent_books,
                spent_clothing=spent_clothing,
                spent_electronics=spent_electronics,
                spent_furniture=spent_furniture,
                spent_groceries=spent_groceries,
                total_interactions=total_interactions,
                resolved_interactions=resolved_interactions,
                resolution_rate=resolution_rate,
                LoginFrequency=login_frequency,
                ServiceUsage=service_usage
            )

            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results, prob = predict_pipeline.predict(pred_df)

            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                churn_status = "Will Churn" if results[0] == 1 else "Will Stay"
                color = "red" if results[0] == 1 else "green"
                st.markdown(f"<h2 style='color: {color}'>{churn_status}</h2>", unsafe_allow_html=True)

            with col2:
                st.metric("Churn Probability", f"{prob[0]:.2%}")

            with col3:
                risk_level = "High" if prob[0] > 0.7 else "Medium" if prob[0] > 0.4 else "Low"
                risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
                st.markdown(f"<h3 style='color: {risk_color}'>Risk: {risk_level}</h3>", unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob[0] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.error("Please make sure the model is trained first by running the training pipeline.")

elif page == "Analytics":
    st.header("📊 Data Analytics Dashboard")

    # Check if data exists
    data_path = "artifacts/data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            churn_rate = df['ChurnStatus'].mean()
            st.metric("Churn Rate", f"{churn_rate:.2%}")
        with col3:
            avg_spent = df['total_spent'].mean()
            st.metric("Avg Total Spent", f"${avg_spent:.2f}")
        with col4:
            avg_interactions = df['total_interactions'].mean()
            st.metric("Avg Interactions", f"{avg_interactions:.1f}")

        # Charts
        st.subheader("📈 Churn Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Churn by Income Level
            churn_by_income = df.groupby('IncomeLevel')['ChurnStatus'].mean().reset_index()
            fig = px.bar(churn_by_income, x='IncomeLevel', y='ChurnStatus',
                        title='Churn Rate by Income Level',
                        labels={'ChurnStatus': 'Churn Rate'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn by Age Group
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=['18-30', '31-50', '51-70', '70+'])
            churn_by_age = df.groupby('AgeGroup')['ChurnStatus'].mean().reset_index()
            fig = px.bar(churn_by_age, x='AgeGroup', y='ChurnStatus',
                        title='Churn Rate by Age Group',
                        labels={'ChurnStatus': 'Churn Rate'})
            st.plotly_chart(fig, use_container_width=True)

        # More detailed analysis
        st.subheader("💰 Spending Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Spending distribution by churn
            fig = px.box(df, x='ChurnStatus', y='total_spent',
                        title='Total Spending Distribution by Churn Status')
            fig.update_xaxis(tickvals=[0, 1], ticktext=['Retained', 'Churned'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Login frequency by churn
            fig = px.box(df, x='ChurnStatus', y='LoginFrequency',
                        title='Login Frequency by Churn Status')
            fig.update_xaxis(tickvals=[0, 1], ticktext=['Retained', 'Churned'])
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Data not found. Please run the training pipeline first to generate the data.")

elif page == "About":
    st.header("ℹ️ About This Application")

    st.markdown('''
    ## Customer Churn Prediction System

    This application uses machine learning to predict customer churn based on various customer attributes and behaviors.

    ### Features:
    - **Predictive Analytics**: Uses advanced ML algorithms to predict churn probability
    - **Interactive Dashboard**: Visualize key metrics and trends
    - **Real-time Predictions**: Get instant predictions for new customers

    ### Data Sources:
    - Customer Demographics (Age, Gender, Marital Status, Income)
    - Transaction History (Spending patterns, frequency, categories)
    - Customer Service Interactions (Resolution rates, interaction frequency)
    - Online Activity (Login frequency, service usage patterns)

    ### Machine Learning Pipeline:
    1. **Data Ingestion**: Collect and merge data from multiple sources
    2. **Data Transformation**: Clean, preprocess, and engineer features
    3. **Model Training**: Train multiple ML models and select the best performer
    4. **Prediction**: Deploy the model for real-time predictions

    ### Technical Stack:
    - **Frontend**: Streamlit
    - **ML Libraries**: Scikit-learn, XGBoost, CatBoost
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly

    ---
    *Built with ❤️ for better customer retention strategies*
    ''')

# Footer
st.markdown("---")
st.markdown("© 2024 Customer Churn Prediction System")
