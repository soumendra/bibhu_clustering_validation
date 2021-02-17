from sklearn.compose import ColumnTransformer  # type:ignore
from sklearn.ensemble import RandomForestRegressor  # type:ignore
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # type:ignore
from sklearn.pipeline import Pipeline  # type:ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type:ignore
import numpy as np
import scipy as sp  # type:ignore

hp_dict = {
    "model__max_features": sp.stats.uniform(0, 1),
    "model__max_samples": sp.stats.uniform(0, 1),
    "model__n_estimators": [
        50,
        100,
        150,
        200,
        250
    ],
    "model__max_depth": [
        6,
        9,
        12,
        15,
        18,
        21,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        3,
        5,
    ],
    "model__n_jobs": [-1],
    "model__random_state": [2021],
}


def train_model(X, y):
    numerical_ix = X.select_dtypes(include=[np.number, "int64", "float64"]).columns
    categorical_ix = X.select_dtypes(include=["object", "bool", "category"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=2021,
            )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_ix),
            ("cat", categorical_transformer, categorical_ix),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor()),
        ]
    )

    random_searcher = RandomizedSearchCV(
        pipeline,
        param_distributions=hp_dict,
        n_iter=50,
        n_jobs=-1,
        cv=3,
        random_state=2021,
        scoring="neg_mean_squared_error",
    )

    try:
        random_searcher.fit(X_train, y_train)
        test_score = random_searcher.score(X_test, y_test)
    except:
        print(f"Model training failed!")

    return random_searcher, test_score


