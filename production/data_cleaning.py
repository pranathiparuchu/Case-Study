"""Data cleaning functions for the files in the raw dataset.

The intention here is to clean the data as we learn more about the datasets
and enforce some of the business rules/constraints.

Step 1: Clean up column names and set explicit datatypes for data fields, 
and save a cleaned version of each of our raw data files.

Step 2: Merge the individual cleaned files to create a single processed attrition file.

Step 3: Split this processed file into features & target for train & test period respectively.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import ta_lib.core.api as dataset
from ta_lib.core.api import register_processor, string_cleaning


@register_processor("data-cleaning", "clean-all-tables")
def loading_and_cleaning_data(context, params):
    """
    Read and clean raw data.
    """

    data = dict()
    # Dictionary to ensure that each area name is mapped to only one area code in all datasets where column asm_area_code is present
    dict_area_code_dup = {
        "BIHAE": "BSHAE",
        "BIHAW": "BSHAW",
        "COIM": "COSM",
        "DELHM": "DEQHM",
        "HUBL": "HUBQ",
        "KOLK": "KOQH",
        "ORIS": "ORSS",
        "VIJA": "VSJA",
    }
    # Load each raw data file and clean strings
    for i in dataset.list_datasets(context, prefix="/raw/"):
        dataset_name = i.replace("/raw/", "")
        key_ = dataset_name + "_df"
        data[key_] = dataset.load_dataset(context, i)

        data[key_].columns = string_cleaning(data[key_].columns, lower=True)

        if key_ == "doj_df":
            data[key_]["date_of_joining"] = pd.to_datetime(
                data[key_]["date_of_joining"]
            )
        elif key_ == "returns_df":
            data[key_] = data[key_].replace("#NAME?", np.nan)
            data[key_] = data[key_].replace("inf", np.nan)
            for float_col in [
                "profit_without_udaan_sub",
                "profit_with_udaan_without_sub",
                "profit_without_udaan_with_sub",
                "profit_with_udaan_with_sub",
                "numerator_without_udaan_sub",
                "numerator_with_udaan_without_sub",
                "numerator_without_udaan_with_sub",
                "numerator_with_udaan_with_sub",
                "roi_without_udaan_sub",
                "roi_with_udaan_without_sub",
                "roi_without_udaan_with_sub",
                "roi_with_udaan_with_sub",
            ]:
                data[key_] = data[key_].change_type(float_col, float)

        if "asm_area_code" in data[key_].columns:
            fil_ = data[key_]["asm_area_code"].isin(dict_area_code_dup.keys())
            data[key_].loc[fil_, "asm_area_code"] = (
                data[key_].loc[fil_, "asm_area_code"].map(dict_area_code_dup)
            )

        if ("year" in data[key_].columns) and ("month" in data[key_].columns):
            data[key_]["month_start_date"] = pd.to_datetime(
                dict(
                    year=data[key_].year,
                    month=data[key_].month,
                    day=[1] * len(data[key_]),
                )
            )

        data[key_].drop_duplicates(inplace=True)

        dataset.save_dataset(context, data[key_], "cleaned/" + dataset_name)


@register_processor("data-cleaning", "clean-churn-dataset")
def churn_dataset_generation(context, params):
    """Create modeling dataset.

    This function is used to load various datasets and
    merge all the datasets for modelling
    """

    data = dict()
    # sai
    for i in dataset.list_datasets(context, prefix="/cleaned/"):
        key_ = i.replace("/cleaned/", "")
        data[key_ + "_df"] = dataset.load_dataset(context, i)

    ref_date_for_churn = params["ref_date"]
    # Exhaustive set of customer ids in the data to be considered for churn
    df_population = (
        data["pri_bpm_df"][["customer_code"]]
        .drop_duplicates()
        .merge(data["doj_df"], on="customer_code", how="left", validate="1:1")
        .query(
            f"date_of_joining <= '{ref_date_for_churn}'or date_of_joining.isnull()",
            engine="python",
        )[["customer_code"]]
    )
    df_population.drop_duplicates(inplace=True)

    # Define attrition start date as the first month where sales become 0 or negative
    cust = (
        data["pri_bpm_df"]
        .query("pri_sales_amount <= 0")
        .groupby("customer_code")["month_start_date"]
        .min()
        .rename("attrition_month_strt")
        .reset_index()
    )

    cust["ref_date"] = pd.to_datetime(ref_date_for_churn)

    cust["days_from_ref_date_to_attrition"] = (
        cust.attrition_month_strt - cust.ref_date
    ).dt.days

    # Set target label=1 for these churned customers
    cust["target"] = 1
    df_population = df_population.merge(cust, on="customer_code", how="left")

    # filtering out customers who have already been churned before the reference date
    df_population = df_population[
        (df_population.days_from_ref_date_to_attrition >= 0)
        | (df_population.days_from_ref_date_to_attrition.isnull())
    ]

    df_population = df_population.drop(
        [
            "ref_date",
            "days_from_ref_date_to_attrition",
            "attrition_month_strt",
        ],
        axis=1,
    )
    df_population["target"].fillna(0, inplace=True)
    df_population["ref_date"] = pd.to_datetime(ref_date_for_churn)

    # Add feature: Customer area code
    df_customer_area_code = pd.DataFrame()
    for key_ in [
        "pri_bpm_df",
        "coverage_df",
        "retail_program_df",
        "ec_df",
        "ordered_with_app_df",
        "ordered_without_app_df",
        "dist_retail_invoice_df",
    ]:
        df_customer_area_code = df_customer_area_code.append(
            data[key_]
            .filter(["customer_code", "asm_area_code"])
            .drop_duplicates()
        ).drop_duplicates()

    df_population = df_population.merge(
        df_customer_area_code, on="customer_code", how="left", validate="1:1"
    )

    # Add feature: Customer's date of joining
    df_population = df_population.merge(
        data["doj_df"], on="customer_code", how="left", validate="1:1"
    )
    dataset.save_dataset(context, df_population, "processed/attrition")


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Generate train & test datasets."""

    df_population = dataset.load_dataset(context, "processed/attrition")
    df_population.set_index("customer_code", inplace=True)
    train_X, test_X, train_y, test_y = train_test_split(
        df_population.drop(params["target_col"], axis=1),
        df_population[params["target_col"]],
        test_size=params["test_size"],
        random_state=context.random_seed,
    )
    train_X = train_X.reset_index()
    test_X = test_X.reset_index()
    train_y = train_y.reset_index()
    test_y = test_y.reset_index()

    print("Saving training datasets")
    dataset.save_dataset(context, train_X, "train/attrition/features")
    dataset.save_dataset(context, train_y, "train/attrition/target")

    print("Saving test datasets")
    dataset.save_dataset(context, test_X, "test/attrition/features")
    dataset.save_dataset(context, test_y, "test/attrition/target")
