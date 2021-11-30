import os.path as op
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from scripts import custom_data_transform
from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_DATA_BASE_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataframe,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train the model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # Load datasets
    train_X = load_dataset(context, "train/attrition/features")
    train_y = load_dataset(context, "train/attrition/target")

    # Load pipelines
    feature_generation_ppln = load_pipeline(
        op.join(artifacts_folder, "feature_generation_ppln.joblib")
    )
    binning_transformer = load_pipeline(
        op.join(artifacts_folder, "binning_transformer.joblib")
    )
    curated_columns = load_pipeline(
        op.join(artifacts_folder, "curated_columns.joblib")
    )
    outlier_transformer = load_pipeline(
        op.join(artifacts_folder, "outlier_transformer.joblib")
    )
    features_transform_ppln = load_pipeline(
        op.join(artifacts_folder, "features_transformer.joblib")
    )

    train_X = feature_generation_ppln.fit_transform(train_X, train_y)
    train_X.set_index("customer_code", inplace=True)
    train_y.set_index("customer_code", inplace=True)

    train_X_binned = get_dataframe(
        binning_transformer.transform(train_X),
        get_feature_names_from_column_transformer(binning_transformer),
    )

    train_X_binned.index = train_X.index
    train_X = train_X_binned.infer_objects()

    train_X = outlier_transformer.transform(train_X)
    train_X = get_dataframe(
        features_transform_ppln.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(features_transform_ppln),
    )
    train_X = train_X[curated_columns + params["imp_features"]]
    rf_pipeline_final = Pipeline(
        [
            (
                "",
                FunctionTransformer(
                    custom_data_transform,
                    kw_args={"cols2keep": params["imp_features"]},
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(**params["grid_best_params"]),
            ),
        ]
    )
    rf_pipeline_final.fit(train_X, train_y)

    save_pipeline(
        rf_pipeline_final,
        op.abspath(op.join(artifacts_folder, "model_pipeline.joblib")),
    )
    print(f"Pipeline saved!")
