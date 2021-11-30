"""Module for listing down additional custom functions required for production."""

import numpy as np
import pandas as pd
import pandas_flavor as pf
from dateutil.relativedelta import relativedelta
from sklearn.base import BaseEstimator, TransformerMixin
from xverse.transformer import WOE

import ta_lib.core.api as dataset


def custom_data_transform(df, cols2keep=[]):
    """Drop columns in the data.

    Parameters
    ----------
        df:  pd.DataFrame
            Dataframe in which requiredcolumns are to be kept
        cols2keep: list
            Columns to keep in the dataframe

    Returns
    -------
        df: pd.DataFrame
    """
    if len(cols2keep):
        return df.select_columns(cols2keep)
    else:
        return df


class CustomFeatureGeneration(BaseEstimator, TransformerMixin):
    """Feature engineering class."""

    def __init__(self, context, ref_date, agg_methods):
        self.context = context
        self.ref_date = ref_date
        self.agg_methods = agg_methods

    def fit(self, X, y=None):
        pass

    def add_features(self, df, context, ref_date, agg_methods):
        data = (
            df.add_time_features(
                context,
                date_col="month_start_date",
                features_col=agg_methods["primary_sales"]["feature_col"],
                data_path="cleaned/pri_bpm",
                ref_date=ref_date,
                agg_type=agg_methods["primary_sales"]["agg_type"],
                windows=agg_methods["primary_sales"]["window"],
                group_by_col=agg_methods["primary_sales"]["group_by_col"],
                agg_method=agg_methods["primary_sales"]["method"],
            )
            .add_time_features(
                context,
                date_col="month_start_date",
                features_col=agg_methods["secondary_sales"]["feature_col"],
                data_path="/cleaned/sec_bpm",
                ref_date=ref_date,
                agg_type=agg_methods["secondary_sales"]["agg_type"],
                windows=agg_methods["secondary_sales"]["window"],
                group_by_col=agg_methods["secondary_sales"]["group_by_col"],
                agg_method=agg_methods["secondary_sales"]["method"],
            )
            .add_features_no_objection_data(context, ref_date)
            .add_time_features(
                context,
                date_col="month_start_date",
                features_col=agg_methods["order_alloc"]["feature_col"],
                data_path="/cleaned/order_alloc",
                ref_date=ref_date,
                agg_type=agg_methods["order_alloc"]["agg_type"],
                windows=agg_methods["order_alloc"]["window"],
                group_by_col=agg_methods["order_alloc"]["group_by_col"],
                agg_method=agg_methods["order_alloc"]["method"],
            )
            .add_time_features(
                context,
                date_col="month_start_date",
                features_col=agg_methods["coverage_data"]["feature_col"],
                data_path="/cleaned/coverage",
                ref_date=ref_date,
                agg_type=agg_methods["coverage_data"]["agg_type"],
                windows=agg_methods["coverage_data"]["window"],
                group_by_col=agg_methods["coverage_data"]["group_by_col"],
                agg_method=agg_methods["coverage_data"]["method"],
            )
            .add_features_profit_data(
                context,
                ref_date,
                agg_type=agg_methods["profit_data"]["agg_type"],
                windows=agg_methods["profit_data"]["window"],
                group_by_col=agg_methods["profit_data"]["group_by_col"],
                agg_method=agg_methods["profit_data"]["method"],
            )
            .add_features_order_alloc_reason(context, ref_date)
            .add_features_retail_program_data(
                context,
                ref_date,
                agg_type=agg_methods["retail_program"]["agg_type"],
                windows=agg_methods["retail_program"]["window"],
                group_by_col=agg_methods["retail_program"]["group_by_col"],
                agg_method=agg_methods["retail_program"]["method"],
            )
            .add_time_features(
                context,
                date_col="month_start_date",
                features_col=agg_methods["business_data"]["feature_col"],
                data_path="/cleaned/ec",
                ref_date=ref_date,
                agg_type=agg_methods["business_data"]["agg_type"],
                windows=agg_methods["business_data"]["window"],
                group_by_col=agg_methods["business_data"]["group_by_col"],
                agg_method=agg_methods["business_data"]["method"],
            )
            .add_time_features(
                context,
                date_col="month_start_date",
                features_col=agg_methods["app_order_data"]["feature_col"],
                data_path="/cleaned/ordered_with_app",
                ref_date=ref_date,
                agg_type=agg_methods["app_order_data"]["agg_type"],
                windows=agg_methods["app_order_data"]["window"],
                group_by_col=agg_methods["app_order_data"]["group_by_col"],
                agg_method=agg_methods["app_order_data"]["method"],
            )
            .add_time_features(
                context,
                date_col="month_start_date",
                features_col=agg_methods["withoutapp_order_data"][
                    "feature_col"
                ],
                data_path="/cleaned/ordered_without_app",
                ref_date=ref_date,
                agg_type=agg_methods["withoutapp_order_data"]["agg_type"],
                windows=agg_methods["withoutapp_order_data"]["window"],
                group_by_col=agg_methods["withoutapp_order_data"][
                    "group_by_col"
                ],
                agg_method=agg_methods["withoutapp_order_data"]["method"],
            )
            .add_features_retail_invoices_data(
                context,
                ref_date,
                agg_type=agg_methods["retail_data"]["agg_type"],
                windows=agg_methods["retail_data"]["window"],
                group_by_col=agg_methods["retail_data"]["group_by_col"],
                agg_method=agg_methods["retail_data"]["method"],
            )
            .add_days_in_business_feature(context, ref_date)
        )
        return data

    def transform(self, X, y=None):
        return self.add_features(
            X,
            self.context,
            self.ref_date,
            self.agg_methods,
        )

    def fit_transform(self, X, y=None):
        return self.add_features(
            X,
            self.context,
            self.ref_date,
            self.agg_methods,
        )


def get_months_data(series, time_col, agg_col, window, method):
    """
    Filter data for required time windows upto to a max date available.

    Parameters
    ----------
    series: pd.DataFrame
        Dataframe consisting of date column and required metric columns
    time_col: str
        Column by which dataframe values are to be sorted by timestamp
    agg_col: str
        Column for which average/lag over required time window is to be calculated
    window: int
        Number of time windows to look back for mean/lag calculations
    method: str
        Any of ['mean', 'sum', 'lag']

    Returns
    -------
    pd.Series
    """

    series = series.sort_values(time_col)
    max_date = series[time_col].max()
    min_date = max_date - relativedelta(months=window)
    if method == "mean":
        # max_date = series[time_col].max()
        # min_date = max_date - relativedelta(months=window)
        val = series[
            (series.month_start_date >= min_date)
            & (series.month_start_date <= max_date)
        ][agg_col].mean()
    elif method == "sum":
        val = series[
            (series.month_start_date >= min_date)
            & (series.month_start_date <= max_date)
        ][agg_col].sum()
    elif method == "lag":
        # max_date = series[time_col].max()
        # min_date = max_date - relativedelta(months=window)
        val = series[(series.month_start_date == min_date)][agg_col]
        if len(val):
            pass
        else:
            val = pd.Series([np.nan], name=agg_col)
    return val


def get_rolling_time_features(
    df, time_col, agg_col, groupbycol, windows, method
):
    """
    Calculate average of a series over a given time window

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe consisting of date column and required metric columns
    time_col: str
        Column by which dataframe values are to be sorted by timestamp
    agg_col: str
        Column for which average over required time window is to be calculated
    groupbycol: str
        Column over which we groupby/aggregate if data hierarchy is to be changed
    windows: list
        Number of time windows to look back for mean/lag calculations
    method: str
        Any of ['mean', 'sum', 'lag']

    Returns
    -------
    pd.DataFrame

    """
    window_dfs = {}
    for w in windows:
        if groupbycol:
            window_dfs["last" + str(w) + "months"] = (
                df.groupby("customer_code")
                .apply(
                    lambda x: get_months_data(
                        x,
                        time_col=time_col,
                        agg_col=agg_col,
                        window=w,
                        method=method,
                    )
                )
                .rename("last_" + str(w) + "_months_avg_" + agg_col)
            )
        else:
            window_dfs["last" + str(w) + "months"] = df.apply(
                lambda x: get_months_data(
                    x,
                    time_col=time_col,
                    agg_col=agg_col,
                    window=w,
                    method=method,
                )
            ).rename("last_" + str(w) + "_months_avg_" + agg_col)
    features = pd.concat([pxm_df for _, pxm_df in window_dfs.items()], axis=1)
    return features


def get_cumulative_time_features(df, agg_col, groupbycol, method):
    """
    Calculate overall cumulative sum/mean across all historical data available

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe consisting of date column and required metric columns
    agg_col: int
        Column for which cumulative sum/average is to be calculated
    groupbycol: str
        Column over which we groupby/aggregate if data hierarchy is to be changed
    method: str
        Either of 'mean' or 'sum'

    Returns
    -------
    pd.Series
    """
    if method == "mean":
        if groupbycol:
            overall = (
                df.groupby(groupbycol)[agg_col]
                .mean()
                .rename("overall_avg_" + agg_col)
            )
        else:
            overall = df[agg_col].mean().rename("overall_avg_" + agg_col)
    elif method == "sum":
        if groupbycol:
            overall = (
                df.groupby(groupbycol)[agg_col]
                .sum()
                .rename("overall_sum_" + agg_col)
            )
        else:
            overall = df[agg_col].sum().rename("overall_sum_" + agg_col)
    return overall


def get_quarter_from_month(dt):
    """Get the quarted for a particular month."""

    mth = dt.month
    quarter = 0
    yr = int(dt.year)
    if mth in [1, 2, 3]:
        yr = yr - 1
        quarter = 4
    elif mth in [4, 5, 6]:
        quarter = 1
    elif mth in [7, 8, 9]:
        quarter = 2
    elif mth in [10, 11, 12]:
        quarter = 3
    return quarter


@pf.register_dataframe_method
def add_time_features(
    df,
    context,
    date_col,
    features_col,
    data_path,
    ref_date,
    agg_type,
    windows,
    group_by_col,
    agg_method,
):
    """
    Addition of Time Features to the data.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe consisting of date column and required metric columns
    date_col:
        Date column to consider for churn
    features_col: str
        Column to transform for time features
    data_path: str
        Path to load data from which time features are to be calculated
    ref_date: str
        Cutoff date to consider for churn
    agg_type: list
        Specify if features should be 'rolling' or 'cumulative' or both
    windows: int
        Number of time windows to look back for feature calculations
    group_by_col: str
        Column over which we groupby/aggregate if data hierarchy is to be changed
    agg_method: str
        Either of 'mean' or 'sum'


    Returns
    -------
    pd.Series
    """
    df = df.copy()
    customers = df[group_by_col].drop_duplicates().tolist()
    cleaned_df = dataset.load_dataset(context, data_path)
    cleaned_df_slice = cleaned_df[(cleaned_df[group_by_col].isin(customers))]
    cleaned_df_slice = cleaned_df_slice[cleaned_df_slice[date_col] <= ref_date]
    if "rolling" in agg_type:
        pxm_avg_df = get_rolling_time_features(
            df=cleaned_df_slice,
            time_col=date_col,
            agg_col=features_col,
            groupbycol=group_by_col,
            windows=windows,
            method=agg_method,
        )
    if "cumulative" in agg_type:
        overall = get_cumulative_time_features(
            cleaned_df_slice,
            agg_col=features_col,
            groupbycol=group_by_col,
            method=agg_method,
        )
    if "rolling" in agg_type and "cumulative" in agg_type:
        features = pd.concat([overall, pxm_avg_df], axis=1).reset_index()
        df = df.merge(features, on=group_by_col, how="left", validate="1:1")
    elif agg_type == ["cumulative"]:
        features = overall.reset_index()
        df = df.merge(features, on=group_by_col, how="left", validate="1:1")
    elif agg_type == ["rolling"]:
        features = pxm_avg_df.reset_index()
        df = df.merge(features, on=group_by_col, how="left", validate="1:1")
    else:
        pass
    # features = features.reset_index()
    return df


@pf.register_dataframe_method
def add_features_no_objection_data(df, context, ref_date):
    """
    Addition of  Feature in to the data.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe consisting of no objection data
    ref_date: str
        Cutoff date to consider for churn

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    customers = df.customer_code.unique().tolist()
    ref_date = pd.to_datetime(ref_date)
    no_obj_df = dataset.load_dataset(context, "/cleaned/no_obj").query(
        "(customer_code in @customers)", engine="python"
    )
    no_obj_df_slice = no_obj_df[
        (no_obj_df.claim_year <= ref_date.year)
        | (
            (no_obj_df.claim_year == ref_date.year)
            & (
                no_obj_df.quarter.str.replace("P", "").astype(int)
                <= get_quarter_from_month(ref_date)
            )
        )
    ]
    features = (
        no_obj_df_slice.groupby(["customer_code", "policy_flag"])
        .size()
        .rename("policy_count")
        .reset_index()
        .pivot(index="customer_code", columns="policy_flag")
        .fillna(0)
    )
    features.columns = [x[0] + "_" + x[1].lower() for x in features.columns]
    features = features.reset_index()
    df = df.merge(features, on="customer_code", how="left", validate="1:1")
    return df


@pf.register_dataframe_method
def add_features_order_alloc_reason(df, context, ref_date):
    """
    Addition of Order Allocation reason to the dataframe.

     Parameters
    ----------
    df: pd.DataFrame
        Dataframe consisting of no objection data
    ref_date: str
        Cutoff date to consider for churn

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    customers = df["customer_code"].drop_duplicates().tolist()
    features = (
        dataset.load_dataset(context, "/cleaned/orders_and_allocated_reason")
        .query(
            f"(customer_code in @customers) & (month_start_date<='{ref_date}')",
            engine="python",
        )
        .groupby(["customer_code", "reason_code_description_new_new"])[
            ["order_qty_in_cases", "allocated_qty_in_cases"]
        ]
        .apply(
            lambda x: x["allocated_qty_in_cases"].sum()
            / x["order_qty_in_cases"].sum()
        )
        .rename("allocated_percent_reason_")
        .reset_index()
        .pivot(
            index="customer_code", columns="reason_code_description_new_new"
        )
        .fillna(0)
    )
    features.columns = [x[0] + "_" + x[1].lower() for x in features.columns]
    features = features.reset_index()
    df = df.merge(features, on="customer_code", how="left", validate="1:1")
    return df


@pf.register_dataframe_method
def add_features_profit_data(
    df, context, ref_date, agg_type, windows, group_by_col, agg_method
):
    """
    Addition of profit features in to the data.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe consisting of date column and required metric columns
    ref_date: str
        Cutoff date to consider for churn
    agg_type: list
        Specify if features should be 'rolling' or 'cumulative' or both
    windows: int
        Number of time windows to look back for feature calculations
    group_by_col: str
        Column over which we groupby/aggregate if data hierarchy is to be changed
    agg_method: str
        Either of 'mean' or 'sum'

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()
    customers = df[group_by_col].drop_duplicates().tolist()
    return_df = dataset.load_dataset(context, "/cleaned/returns")
    return_df = return_df.replace([np.inf, -np.inf], np.nan)
    cleaned_df_slice = return_df[(return_df.customer_code.isin(customers))]
    cleaned_df_slice = cleaned_df_slice[
        cleaned_df_slice.month_start_date <= ref_date
    ]
    features = pd.DataFrame(
        {
            "customer_code": cleaned_df_slice[group_by_col]
            .drop_duplicates()
            .tolist()
        }
    )

    for i in [
        "profit_with_udaan_without_sub",
        "profit_without_udaan_with_sub",
        "profit_with_udaan_with_sub",
        "roi_without_udaan_sub",
        "roi_with_udaan_without_sub",
        "roi_without_udaan_with_sub",
        "roi_with_udaan_with_sub",
    ]:
        if "rolling" in agg_type:
            pxm_avg_df = get_rolling_time_features(
                df=cleaned_df_slice,
                time_col="month_start_date",
                agg_col=i,
                groupbycol="customer_code",
                windows=windows,
                method=agg_method,
            )
        if "cumulative" in agg_type:
            overall = get_cumulative_time_features(
                cleaned_df_slice,
                agg_col=i,
                groupbycol=group_by_col,
                method=agg_method,
            )
            atemp = pd.concat([overall, pxm_avg_df], axis=1)
        else:
            atemp = pxm_avg_df
        features = features.merge(
            atemp, on="customer_code", how="left", validate="1:1"
        )
    df = df.merge(features, on="customer_code", how="left", validate="1:1")
    return df


@pf.register_dataframe_method
def add_features_retail_program_data(
    df, context, ref_date, agg_type, windows, group_by_col, agg_method
):
    """
    Addition of retail program Feature in to the data.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe consisting of date column and required metric columns
    ref_date: str
        Cutoff date to consider for churn
    agg_type: list
        Specify if features should be 'rolling' or 'cumulative' or both
    windows: int
        Number of time windows to look back for feature calculations
    group_by_col: str
        Column over which we groupby/aggregate if data hierarchy is to be changed
    agg_method: str
        Either of 'mean' or 'sum'

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    customers = df.customer_code.unique().tolist()
    retail_program_df = dataset.load_dataset(
        context, "/cleaned/retail_program"
    )
    retail_program_df_slice = retail_program_df[
        (retail_program_df.customer_code.isin(customers))
    ]
    retail_program_df_slice = retail_program_df_slice[
        retail_program_df_slice.month_start_date <= ref_date
    ]
    features = pd.DataFrame(
        {
            "customer_code": retail_program_df_slice.customer_code.unique().tolist()
        }
    )

    for i in [
        "sec_netvalue",
        "bandhan_net_val",
        "wholesale_net_value",
        "otherpgrm_net_val",
        "non_wholesale_net_value",
    ]:
        if "rolling" in agg_type:
            pxm_avg_df = get_rolling_time_features(
                df=retail_program_df_slice,
                time_col="month_start_date",
                agg_col=i,
                groupbycol="customer_code",
                windows=windows,
                method=agg_method,
            )
        if "cumulative" in agg_type:
            overall = get_cumulative_time_features(
                retail_program_df_slice,
                groupbycol=group_by_col,
                agg_col=i,
                method=agg_method,
            )
            atemp = pd.concat([overall, pxm_avg_df], axis=1)
        else:
            atemp = pxm_avg_df
        features = features.merge(
            atemp, on="customer_code", how="left", validate="1:1"
        )

    df = df.merge(features, on="customer_code", how="left", validate="1:1")
    return df


@pf.register_dataframe_method
def add_features_retail_invoices_data(
    df, context, ref_date, agg_type, windows, group_by_col, agg_method
):
    """
    Addition of retail invoices feature to the data.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe consisting of date column and required metric columns
    ref_date: str
        Cutoff date to consider for churn
    agg_type: list
        Specify if features should be 'rolling' or 'cumulative' or both
    windows: int
        Number of time windows to look back for feature calculations
    group_by_col: str
        Column over which we groupby/aggregate if data hierarchy is to be changed
    agg_method: str
        Either of 'mean' or 'sum'

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    customers = df.customer_code.unique().tolist()
    retail_invoices_df = dataset.load_dataset(
        context, "/cleaned/dist_retail_invoice"
    )
    retail_invoices_df_slice = retail_invoices_df[
        (retail_invoices_df.customer_code.isin(customers))
    ]
    retail_invoices_df_slice = retail_invoices_df_slice[
        retail_invoices_df_slice.month_start_date <= ref_date
    ]
    features = pd.DataFrame(
        {
            "customer_code": retail_invoices_df_slice.customer_code.unique().tolist()
        }
    )
    for i in ["invoice_count", "unique_retailers"]:
        if "rolling" in agg_type:
            pxm_avg_df = get_rolling_time_features(
                df=retail_invoices_df_slice,
                time_col="month_start_date",
                agg_col=i,
                groupbycol="customer_code",
                windows=windows,
                method=agg_method,
            )
        if "cumulative" in agg_type:
            overall = get_cumulative_time_features(
                retail_invoices_df_slice,
                groupbycol=group_by_col,
                agg_col=i,
                method=agg_method,
            )
            atemp = pd.concat([overall, pxm_avg_df], axis=1)
        else:
            atemp = pxm_avg_df
        features = features.merge(
            atemp, on="customer_code", how="left", validate="1:1"
        )
    df = df.merge(features, on="customer_code", how="left", validate="1:1")
    return df


@pf.register_dataframe_method
def add_days_in_business_feature(df, context, ref_date):
    """Addition of days in business feature to the data."""
    df = df.copy()
    customers = df.customer_code.unique().tolist()
    features = (
        dataset.load_dataset(context, "/cleaned/doj")
        .query("customer_code in @customers", engine="python")
        .set_index("customer_code")["date_of_joining"]
        .apply(lambda x: (pd.to_datetime(ref_date) - x).days)
        .rename("days_in_business")
        .reset_index()
    )
    df = df.merge(features, on="customer_code", how="left", validate="1:1")
    return df


def woe_binning(x_train, x_test, y_train, col_bin):
    """
    Bin numeric variables using WOE(weight of evidence).

    Parameters
    ----------
    x_train: pd.DataFrame
        Training dataset
    x_test: pd.DataFrame
        Test dataset
    y_train: pd.DataFrame
        Target vairble of training dataset. (Binary/Dichotomous)
    col_bin : str
        Numeric variable column to bucket.

    Returns
    -------
    pd.DataFrame
    """

    woe_binning = WOE(treat_missing="mode").fit(x_train[[col_bin]], y_train)
    x_train["WOE"] = woe_binning.transform(x_train[[col_bin]])
    x_test["WOE"] = woe_binning.transform(x_test[[col_bin]])
    df_bin = woe_binning.woe_df[["WOE", "Category"]].rename(
        columns={"Category": col_bin + "_woebinning"}
    )
    x_train = x_train.merge(df_bin, on="WOE").drop("WOE", axis=1)
    x_test = x_test.merge(df_bin, on="WOE").drop("WOE", axis=1)
    x_test.drop(col_bin, axis=1, inplace=True)
    x_train.drop(col_bin, axis=1, inplace=True)
    return x_train, x_test
