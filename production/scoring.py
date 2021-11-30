"""Processors for the model scoring/evaluation step of the worklow."""
import os.path as op

import ta_lib.core.api as dataset
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
)


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """Score a pre-trained model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # Load test datasets
    test_X = load_dataset(context, "test/attrition/features")

    # Load the feature pipeline and training pipelines
    feature_generation_ppln = load_pipeline(
        op.join(artifacts_folder, "feature_generation_ppln.joblib")
    )
    binning_transformer = load_pipeline(
        op.join(artifacts_folder, "binning_transformer.joblib")
    )
    curated_columns = load_pipeline(
        op.join(artifacts_folder, "curated_columns.joblib")
    )
    features_transform_ppln = load_pipeline(
        op.join(artifacts_folder, "features_transformer.joblib")
    )
    model_pipeline = load_pipeline(
        op.join(artifacts_folder, "model_pipeline.joblib")
    )

    # Transform the test dataset
    test_X = feature_generation_ppln.transform(test_X)
    customer_codes = test_X.customer_code.tolist()
    test_X.set_index("customer_code", inplace=True)

    test_X_binned = get_dataframe(
        binning_transformer.transform(test_X),
        get_feature_names_from_column_transformer(binning_transformer),
    )

    test_X_binned.index = test_X.index
    test_X = test_X_binned.infer_objects()

    test_X = get_dataframe(
        features_transform_ppln.transform(test_X),
        get_feature_names_from_column_transformer(features_transform_ppln),
    )

    test_X.index = customer_codes
    test_X = test_X[curated_columns + params["imp_features"]]

    # Make a prediction
    test_X["yhat"] = model_pipeline.predict(test_X)

    # Store the predictions for any further processing.
    dataset.save_dataset(context, test_X[["yhat"]], "score/attrition/output")
    print("Saved scoring results!")
