"""Feature Engineering functions for various activities like outlier treatment, encoding, missing value imputation.

The intention here is to have a Pipeline and not model. Our focus is to set it up in such a way that
it can be saved/loaded, tweaked for different model choices and so on.

"""
import os
import os.path as op
import panel as pn
from category_encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import ta_lib.core.api as dataset
import ta_lib.eda.api as ta_analysis
from scripts import CustomFeatureGeneration
from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_DATA_BASE_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataframe,
    load_dataset,
    register_processor,
    save_pipeline,
)
from ta_lib.data_processing.api import Outlier, WoeBinningTransformer

pn.extension("bokeh")


os.environ["TA_DEBUG"] = "False"
os.environ["TA_ALLOW_EXCEPTIONS"] = "True"


def _columns_to_drop(
    corr_table, features_to_be_dropped=[], corr_threshold=0.6
):
    """List features with low correlation."""

    corr_table = corr_table.sort_values("Variable 1")
    for index, row in corr_table.iterrows():
        if row["Abs Corr Coef"] > corr_threshold:
            if row["Variable 1"] not in features_to_be_dropped:
                features_to_be_dropped.append(row["Variable 2"])
    return features_to_be_dropped


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Perform feature transformation and outlier treatment."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = dataset.load_dataset(context, "train/attrition/features")
    train_y = dataset.load_dataset(context, "train/attrition/target")
    test_X = dataset.load_dataset(context, "test/attrition/features")
    test_y = dataset.load_dataset(context, "test/attrition/target")

    ref_date_for_churn = params["ref_date"]
    agg_methods = params["agg_methods"]

    # NOTE: You can use ``Pipeline`` to compose a collection of transformers
    # into a single transformer. In this case, we are composing a
    # customerFeatureGeneration class for feature engineering

    feature_generation_ppln = Pipeline(
        [
            (
                "custom_features_generation",
                CustomFeatureGeneration(
                    context=context,
                    ref_date=ref_date_for_churn,
                    agg_methods=agg_methods,
                ),
            ),
        ]
    )

    train_X = feature_generation_ppln.fit_transform(train_X, train_y)
    test_X = feature_generation_ppln.transform(test_X)
    train_X.set_index("customer_code", inplace=True)
    test_X.set_index("customer_code", inplace=True)
    train_y.set_index("customer_code", inplace=True)
    test_y.set_index("customer_code", inplace=True)

    bin_columns = [
        "overall_avg_pri_sales_amount",
        "overall_avg_roi_without_udaan_with_sub",
    ]
    binning_transformer = ColumnTransformer(
        [("binn", WoeBinningTransformer(encode="onehot"), bin_columns)],
        remainder="passthrough",
    )

    train_X_binned = get_dataframe(
        binning_transformer.fit_transform(train_X, train_y["target"]),
        get_feature_names_from_column_transformer(binning_transformer),
    )

    train_X_binned.index = train_X.index
    train_X = train_X_binned.infer_objects()

    outlier_transformer = Outlier(method="percentile")
    train_X = outlier_transformer.fit_transform(train_X)
    train_y = train_y.loc[train_X.index]

    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    feature_transformation_ppln = ColumnTransformer(
        [
            (
                "onehot_encoding",
                OneHotEncoder(use_cat_names=True),
                cat_columns,
            ),
            (
                "simple_imputation_constant",
                SimpleImputer(strategy="constant", fill_value=0),
                list(set(num_columns) - set(["days_in_business"])),
            ),
            (
                "simple_imputation_median",
                SimpleImputer(strategy="median"),
                ["days_in_business"],
            ),
        ]
    )

    train_X = get_dataframe(
        feature_transformation_ppln.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(feature_transformation_ppln),
    )

    corr_df = ta_analysis.get_correlation_table(train_X[num_columns])
    corr_df_drop = corr_df[
        corr_df["Abs Corr Coef"] > params["correlation_threshold"]
    ]
    columns_to_be_dropped = ["ref_date"]
    columns_to_be_dropped = _columns_to_drop(
        corr_df,
        features_to_be_dropped=columns_to_be_dropped,
        corr_threshold=0.6,
    )
    curated_columns = list(set(train_X.columns) - set(columns_to_be_dropped))

    save_pipeline(
        curated_columns,
        op.abspath(op.join(artifacts_folder, "curated_columns.joblib")),
    )
    save_pipeline(
        feature_generation_ppln,
        op.abspath(
            op.join(artifacts_folder, "feature_generation_ppln.joblib")
        ),
    )
    save_pipeline(
        binning_transformer,
        op.abspath(op.join(artifacts_folder, "binning_transformer.joblib")),
    )
    save_pipeline(
        outlier_transformer,
        op.abspath(op.join(artifacts_folder, "outlier_transformer.joblib")),
    )
    save_pipeline(
        feature_transformation_ppln,
        op.abspath(op.join(artifacts_folder, "features_transformer.joblib")),
    )
    print("Finished Saving Pipelines")
