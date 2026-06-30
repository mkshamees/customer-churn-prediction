import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = joblib.load("../models/customer_churn_model.pkl")

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

body{
    background:#f4f6f9;
}

.main{
    background-color:#f4f6f9;
}

.block-container{
    padding-top:2rem;
}

h1,h2,h3{
    color:#003366;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:20px;
    box-shadow:0px 5px 12px rgba(0,0,0,0.12);
}

.stButton>button{
    width:100%;
    background:#003366;
    color:white;
    border-radius:10px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0055aa;
    color:white;
}

section[data-testid="stSidebar"]{
    background:#0B1F3A;
}

section[data-testid="stSidebar"] *{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("📊 Customer Churn Prediction Dashboard")

st.write(
"""
### Machine Learning Powered Customer Retention System
Predict customers who are likely to leave your company before they churn.
"""
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Dashboard",

        "🔮 Prediction",

        "📊 Analytics",

        "📈 Model Interpretation",

        "💼 Business Insights"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.info("""

Models Available

• Logistic Regression

• Decision Tree

• Random Forest

• Support Vector Machine

• XGBoost

""")

# =====================================================
# DASHBOARD
# =====================================================

if page=="🏠 Dashboard":

    st.header("Dashboard Overview")

    col1,col2,col3,col4=st.columns(4)

    with col1:

        st.metric(

            "Total Customers",

            "7,010"

        )

    with col2:

        st.metric(

            "Models",

            "5"

        )

    with col3:

        st.metric(

            "Best Accuracy",

            "80.81%"

        )

    with col4:

        st.metric(

            "Best ROC AUC",

            "82.17%"

        )

    st.markdown("---")

    model_df=pd.DataFrame({

        "Model":[

            "Logistic Regression",

            "Support Vector Machine",

            "Random Forest",

            "XGBoost",

            "Decision Tree"

        ],

        "Accuracy":[

            0.808,

            0.800,

            0.786,

            0.777,

            0.718

        ]

    })

    fig=px.bar(

        model_df,

        x="Model",

        y="Accuracy",

        color="Accuracy",

        text="Accuracy"

    )

    fig.update_layout(

        title="Model Accuracy Comparison",

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    roc_df=pd.DataFrame({

        "Model":[

            "Logistic Regression",

            "Support Vector Machine",

            "Random Forest",

            "XGBoost",

            "Decision Tree"

        ],

        "ROC AUC":[

            0.722,

            0.777,

            0.685,

            0.822,

            0.635

        ]

    })

    fig2=px.line(

        roc_df,

        x="Model",

        y="ROC AUC",

        markers=True

    )

    fig2.update_layout(

        title="ROC AUC Comparison",

        height=450

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    st.subheader("Model Performance")

    st.dataframe(

        pd.DataFrame({

            "Model":[

                "Logistic Regression",

                "Support Vector Machine",

                "Random Forest",

                "XGBoost",

                "Decision Tree"

            ],

            "Accuracy":[

                "80.81%",

                "80.03%",

                "78.60%",

                "77.75%",

                "71.75%"

            ],

            "ROC AUC":[

                "72.20%",

                "77.75%",

                "68.45%",

                "82.17%",

                "63.45%"

            ]

        }),

        use_container_width=True

    )

    left,right=st.columns(2)

    with left:

        st.success("""

### Key Findings

• Month-to-month contracts have the highest churn.

• Long-term contracts greatly reduce churn.

• Higher monthly charges increase churn.

• Short-tenure customers are the highest-risk group.

""")

    with right:

        st.warning("""

### Business Recommendations

• Target month-to-month customers.

• Improve Fiber Optic customer support.

• Offer loyalty discounts.

• Launch proactive retention campaigns.

""")
        

        
        # =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🔮 Prediction":

    st.header("🔮 Customer Churn Prediction")

    st.write(
        "Fill in the customer information below and click **Predict Churn**."
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        Partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        Dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        PhoneService = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        MultipleLines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        InternetService = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        OnlineSecurity = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        OnlineBackup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        DeviceProtection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        TechSupport = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        StreamingTV = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        StreamingMovies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        Contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )

        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        MonthlyCharges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        TotalCharges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

    st.markdown("---")

    predict_button = st.button(
        "🚀 Predict Customer Churn",
        use_container_width=True
    )
    if predict_button:

        input_df = pd.DataFrame({

            "gender":[gender],

            "SeniorCitizen":[SeniorCitizen],

            "Partner":[Partner],

            "Dependents":[Dependents],

            "tenure":[tenure],

            "PhoneService":[PhoneService],

            "MultipleLines":[MultipleLines],

            "InternetService":[InternetService],

            "OnlineSecurity":[OnlineSecurity],

            "OnlineBackup":[OnlineBackup],

            "DeviceProtection":[DeviceProtection],

            "TechSupport":[TechSupport],

            "StreamingTV":[StreamingTV],

            "StreamingMovies":[StreamingMovies],

            "Contract":[Contract],

            "PaperlessBilling":[PaperlessBilling],

            "PaymentMethod":[PaymentMethod],

            "MonthlyCharges":[MonthlyCharges],

            "TotalCharges":[TotalCharges]

                })

        # Convert binary text values to numeric values
        binary_map = {
            "Female": 0,
            "Male": 1,
            "No": 0,
            "Yes": 1
        }

        input_df["gender"] = input_df["gender"].map(binary_map)
        input_df["Partner"] = input_df["Partner"].map(binary_map)
        input_df["Dependents"] = input_df["Dependents"].map(binary_map)
        input_df["PhoneService"] = input_df["PhoneService"].map(binary_map)
        input_df["PaperlessBilling"] = input_df["PaperlessBilling"].map(binary_map)

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        st.subheader("Prediction Result")

        metric1,metric2=st.columns(2)

        with metric1:

            st.metric(

                "Churn Probability",

                f"{probability:.2%}"

            )

        with metric2:

            st.metric(

                "Retention Probability",

                f"{1-probability:.2%}"

            )

        st.progress(float(probability))

        if prediction==1:

            st.error(

                "⚠ Customer is likely to CHURN."

            )

        else:

            st.success(

                "✅ Customer is likely to STAY."

            )

        if probability>=0.80:

            risk="Very High"

            color="🔴"

        elif probability>=0.60:

            risk="High"

            color="🟠"

        elif probability>=0.40:

            risk="Medium"

            color="🟡"

        else:

            risk="Low"

            color="🟢"

        st.subheader("Customer Risk Level")

        st.info(

            f"{color} {risk}"

        )

        st.markdown("---")

        st.subheader("Customer Summary")

        st.dataframe(

            input_df,

            use_container_width=True

        )

        st.markdown("---")

        st.subheader("Business Recommendation")

        if prediction==1:

            st.warning("""

### Recommended Actions

• Contact customer immediately

• Offer loyalty discount

• Assign retention specialist

• Review service complaints

• Recommend long-term contract

""")

        else:

            st.success("""

### Recommended Actions

• Continue engagement

• Reward customer loyalty

• Promote premium services

• Encourage referrals

""")

        report=input_df.copy()

        report["Prediction"]=[

            "Churn"

            if prediction==1

            else

            "Stay"

        ]

        report["Churn Probability"]=[

            round(probability*100,2)

        ]

        csv=report.to_csv(

            index=False

        ).encode("utf-8")

        st.download_button(

            label="📥 Download Prediction Report",

            data=csv,

            file_name="customer_prediction.csv",

            mime="text/csv"

        )

        st.balloons()


        # =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == "📊 Analytics":

    st.header("📊 Model Analytics")

    st.write(
        "Performance comparison of all machine learning models."
    )

    comparison = pd.DataFrame({

        "Model":[

            "Logistic Regression",

            "Support Vector Machine",

            "Random Forest",

            "XGBoost",

            "Decision Tree"

        ],

        "Accuracy":[

            0.808,

            0.800,

            0.786,

            0.777,

            0.718

        ],

        "Precision":[

            0.671,

            0.669,

            0.628,

            0.595,

            0.466

        ],

        "Recall":[

            0.539,

            0.485,

            0.469,

            0.499,

            0.458

        ],

        "F1 Score":[

            0.598,

            0.563,

            0.537,

            0.543,

            0.462

        ],

        "ROC AUC":[

            0.722,

            0.777,

            0.685,

            0.822,

            0.635

        ]

    })

    st.subheader("Model Comparison")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Accuracy Comparison")

    fig = px.bar(

        comparison,

        x="Model",

        y="Accuracy",

        color="Accuracy",

        text="Accuracy",

        color_continuous_scale="Blues"

    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("ROC AUC Comparison")

    fig = px.line(

        comparison,

        x="Model",

        y="ROC AUC",

        markers=True

    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Performance Heatmap")

    heatmap = px.imshow(

        comparison.set_index("Model"),

        text_auto=True,

        aspect="auto",

        color_continuous_scale="Viridis"

    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        fig = px.pie(

            comparison,

            values="Accuracy",

            names="Model",

            title="Accuracy Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.scatter(

            comparison,

            x="Recall",

            y="Precision",

            color="Model",

            size="Accuracy",

            hover_name="Model"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("🏆 Best Performing Model")

    best = comparison.sort_values(

        by="Accuracy",

        ascending=False

    ).iloc[0]

    st.success(f"""

### {best['Model']}

Accuracy : **{best['Accuracy']:.3f}**

Precision : **{best['Precision']:.3f}**

Recall : **{best['Recall']:.3f}**

F1 Score : **{best['F1 Score']:.3f}**

ROC AUC : **{best['ROC AUC']:.3f}**

""")
    
   # =====================================================
# FEATURE INTERPRETATION (LOGISTIC REGRESSION)
# =====================================================

elif page == "📈 Model Interpretation":

    st.header("🌳 Model Interpretation")

    st.write(
        """
        This page explains how the deployed Logistic Regression model makes
        predictions. Features with positive coefficients increase the likelihood
        of churn, while negative coefficients reduce it.
        """
    )

    try:

        # Get classifier
        classifier = model.named_steps["classifier"]

        # Get feature names after preprocessing
        feature_names = model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        # Clean feature names for better display
        clean_names = []

        for name in feature_names:

            name = name.replace("num__", "")

            name = name.replace("cat__", "")

            name = name.replace("_", " ")

            clean_names.append(name)

        # Logistic Regression coefficients
        coefficients = classifier.coef_[0]

        coef_df = pd.DataFrame({

            "Feature": clean_names,

            "Coefficient": coefficients

        })

        coef_df["Absolute"] = coef_df["Coefficient"].abs()

        coef_df = coef_df.sort_values(
            by="Absolute",
            ascending=False
        )

        st.subheader("Top 20 Most Influential Features")

        st.dataframe(
            coef_df.head(20),
            use_container_width=True
        )

        fig = px.bar(

            coef_df.head(20),

            x="Coefficient",

            y="Feature",

            orientation="h",

            color="Coefficient",

            color_continuous_scale="RdBu",

            text="Coefficient"

        )

        fig.update_layout(

            height=700,

            title="Most Influential Features",

            yaxis={"categoryorder": "total ascending"}

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🔺 Top Features Increasing Churn")

            positive = coef_df.sort_values(

                by="Coefficient",

                ascending=False

            ).head(10)

            st.dataframe(

                positive[["Feature", "Coefficient"]],

                use_container_width=True

            )

        with col2:

            st.subheader("🟢 Top Features Reducing Churn")

            negative = coef_df.sort_values(

                by="Coefficient"

            ).head(10)

            st.dataframe(

                negative[["Feature", "Coefficient"]],

                use_container_width=True

            )

        st.markdown("---")

        st.subheader("Understanding the Results")

        st.info("""

### Positive Coefficients

A positive coefficient means that the feature **increases the probability of customer churn**.

Examples may include:

• Month-to-month contract

• Fiber optic internet

• High monthly charges

---

### Negative Coefficients

A negative coefficient means that the feature **reduces the probability of churn**.

Examples may include:

• Long customer tenure

• Two-year contract

• Tech Support subscription

• Online Security service

---

### Interpretation

The larger the absolute coefficient value, the stronger its influence on the prediction.

""")

        st.markdown("---")

        st.subheader("Model Summary")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(

                "Model",

                "Logistic Regression"

            )

        with c2:

            st.metric(

                "Accuracy",

                "80.81%"

            )

        with c3:

            st.metric(

                "ROC AUC",

                "72.20%"

            )

        st.markdown("---")

        st.success("""

### Business Interpretation

The model indicates that customers are more likely to churn when they:

• Have month-to-month contracts.

• Pay higher monthly charges.

• Subscribe to Fiber Optic internet.

• Have shorter tenure.

Customers are less likely to churn when they:

• Stay with the company for a long time.

• Use two-year contracts.

• Subscribe to Tech Support.

• Subscribe to Online Security.

These insights can help the business target high-risk customers with personalized retention strategies.

""")

    except Exception as e:

        st.error(

            f"Unable to display model interpretation.\n\n{e}"

        )

# =====================================================
# BUSINESS INSIGHTS PAGE
# =====================================================

elif page == "💼 Business Insights":

    st.header("💼 Business Insights")

    c1, c2 = st.columns(2)

    with c1:

        st.success("""

### Key Findings

- Customers with **Month-to-month** contracts churn the most.

- Customers with **Fiber Optic** service are more likely to leave.

- Customers with **short tenure** have higher churn.

- Higher **Monthly Charges** increase churn risk.

- Customers without **Online Security** and **Tech Support**
  have higher churn probability.

""")

    with c2:

        st.warning("""

### Recommended Actions

- Offer long-term contracts.

- Provide loyalty rewards.

- Improve customer support.

- Monitor high-risk customers monthly.

- Target new customers with retention campaigns.

""")

    st.markdown("---")

    st.subheader("Business KPI Dashboard")

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(

            "Customers",

            "7,010"

        )

    with k2:

        st.metric(

            "Best Accuracy",

            "80.81%"

        )

    with k3:

        st.metric(

            "Best ROC AUC",

            "82.17%"

        )

    with k4:

        st.metric(

            "Models Evaluated",

            "5"

        )

    st.markdown("---")

    strategy = pd.DataFrame({

        "Priority":[

            "High",

            "High",

            "Medium",

            "Medium",

            "Low"

        ],

        "Action":[

            "Contact High-Risk Customers",

            "Offer Contract Upgrade",

            "Improve Technical Support",

            "Provide Loyalty Discounts",

            "Upsell Premium Services"

        ],

        "Expected Impact":[

            "Very High",

            "High",

            "Medium",

            "Medium",

            "Low"

        ]

    })

    st.subheader("Retention Strategy")

    st.dataframe(

        strategy,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Expected Business Impact")

    impact = pd.DataFrame({

        "Category":[

            "Customer Retention",

            "Revenue Growth",

            "Customer Satisfaction",

            "Operational Efficiency"

        ],

        "Impact":[

            90,

            82,

            86,

            79

        ]

    })

    fig = px.bar(

        impact,

        x="Category",

        y="Impact",

        color="Impact",

        text="Impact",

        color_continuous_scale="Greens"

    )

    fig.update_layout(

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    """
**Customer Churn Prediction Dashboard**

Developed with **Python**, **Streamlit**, **Scikit-learn**, **XGBoost**, **Plotly**, **Pandas**, and **NumPy**.

© 2026
"""
)