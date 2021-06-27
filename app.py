import streamlit as st
import pandas as pd
import time
from model import train_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# st.markdown(
#     """
# <style>
# .reportview-container .markdown-text-container {
#     font-family: monospace;
# }
# .sidebar .sidebar-content {
#     background-image: linear-gradient(#2e7bcf,#2e7bcf);
#     color: white;
# }
# .Widget>label {
#     color: white;
#     font-family: monospace;
# }
# [class^="st-b"]  {
#     color: white;
#     font-family: monospace;
# }
# .st-bb {
#     background-color: transparent;
# }
# .st-at {
#     background-color: #0c0080;
# }
# footer {
#     font-family: monospace;
# }
# .reportview-container .main footer, .reportview-container .main footer a {
#     color: #0c0080;
# }
# header .decoration {
#     background-image: none;
# }

# </style>
# """,
#     unsafe_allow_html=True,
# )


st.sidebar.markdown("# automl v1")

st.sidebar.markdown("## Upload data")
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type="csv", key="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.sidebar.write(f"{df.shape[0]} rows and {df.shape[1]} columns")


mode = st.sidebar.radio(
    label="",
    options=["Exploratory Data Analysis", "Model training", "Model evaluation"],
    index=0,
)

if mode == "Exploratory Data Analysis":
    if uploaded_csv:
        st.write(df)

if mode == "Model training":
    if uploaded_csv:
        training_history = st.empty()
        st.markdown("<hr />", unsafe_allow_html=True)

        col1, col2 = st.beta_columns(2)
        with col1:
            y_var = st.selectbox(
                label="Choose target variable for prediction",
                options=df.columns,
                index=len(df.columns)-1,
            )

        with col2:
            x_vars = st.multiselect(
                label="Choose dependent variables to be used for modeling",
                options=df.columns,
            )

        X = df.filter(x_vars)
        y = df.loc[: , y_var]

        st.write(f"x_vars: {x_vars}")
        st.write(f"y_var: {y_var}")

        if st.button('Start training'):
            start = time.time()
            results = train_model(X, y)
            end = time.time()
            st.write(f"time_elapsed: {end - start}")
            st.write(f"best_score: {results['model'].best_score_}")
            st.write(f"best_params_: {results['model'].best_params_}")
            st.write(f"test_score: {results['test_score']}")
            st.markdown("<hr />", unsafe_allow_html=True)

            preds = results["model"].predict(results["X_test"])
            outcome = pd.DataFrame({
                "preds": preds,
                "y_test": results["y_test"],
            })
            outcome.loc[:, "dev_ration"] = abs(outcome.preds - outcome.y_test)/outcome.y_test
            st.write(outcome)
            
            st.pyplot(plt.scatter(outcome["y_test"], outcome["preds"], c="blue"))

            forest = results["model"].best_estimator_["model"]
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]

            print("Feature ranking:")

            for f in range(X.shape[1]):
                print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

            fig, ax = plt.subplots()
            # plt.figure()
            plt.title("Feature importances")
            plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
            plt.xticks(range(X.shape[1]), indices)
            plt.xlim([-1, X.shape[1]])
            st.pyplot(fig)

            # r = permutation_importance(forest, results["X_test"], results["y_test"], n_repeats=300, random_state=0)
            # st.write(r.importances_mean)
            # for i in r.importances_mean:
            #     # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            #         st.write(f"{x_vars[i]} \t {r.importances_mean[i]:.3f} \t  +/- {r.importances_std[i]:.3f}")


if mode == "Model evaluation":
    if uploaded_csv:
        st.write(results["model"].best_params_)
