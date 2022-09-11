import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import nevergrad as ng
from concurrent import futures
import time

def make_df_from_sheet(file):
    return pd.read_csv(file)

def preprocess(df):
    '''
    This is a basic preprocessor.
    There may be much better preprocessors built later.
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] # only use numeric types
    df["Classification"] = df["Classification"].str.strip() # strip whitespaces

    # some specific columns to handle
    watched_columns = ["p_astro","SNR_network_matched_filter", "luminosity_distance", "far","chirp_mass_source"]
    for c in watched_columns:
        df = df.fillna(0)

    y = df["Classification"] # the label for classification
    ids = df["id (Event ID)"] # the ids we will use

    df.drop(['far_lower', 'far_upper','p_astro_lower','p_astro_upper'], inplace=True, axis=1) # these were bad columns

    subdf = df.select_dtypes(include=numerics) # only take numeric data for processing
    invalid_columns = subdf.isna().any()[subdf.isna().any().values== True].index
    subdf = subdf.drop(invalid_columns, axis=1)
    subdf = subdf.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)
    ids = ids.reset_index(drop=True)
    return subdf, y, df, ids

def preprocess2(data):
    df = pd.concat([data['Identity'], data['Order'], data['Sales'], data['Product'], data['Location']], axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    df["Categories"] = OneHotEncoder().fit_transform(df[["Category"]]).toarray().astype(int).tolist()
    df["SubCategories"] = OneHotEncoder().fit_transform(df[["Sub-Category"]]).toarray().astype(int).tolist()
    df["Ship Mode Enc"] = OneHotEncoder().fit_transform(df[["Ship Mode"]]).toarray().astype(int).tolist()
    df["Region"] = OneHotEncoder().fit_transform(df[["Region"]]).toarray().astype(int).tolist()
    df["State"] = OneHotEncoder().fit_transform(df[["State"]]).toarray().astype(int).tolist()

    rows = []

    def combine(rows):
        cat_embedding = None
        for row in rows:
            if cat_embedding is None:
                cat_embedding = np.array(row).astype(int)
            else:
                cat_embedding = np.add(cat_embedding, np.array(row).astype(int))
        return cat_embedding

    columns = ["order_id", "total_sales", "discount", "total_quantity", "total_profit", "postal_code", "cateogry_embedding", "subcategory_embedding", "returns", "ship_mode_embedding", "country", "state", "region", "city"]

    for k, v in df.groupby(['Order ID']):
        total_sales_volume = v["Sales"].sum()
        total_quantity = v["Quantity"].sum()
        total_profit = v["Profit"].sum()
        postal_code = v["Postal Code"].iloc[0]
        country = v["Country"].iloc[0]
        state = v["State"].iloc[0]
        region = v["Region"].iloc[0]
        city = v["City"].iloc[0]
        total_discount = v["Discount"].sum()

        cat_embedding = None
        category_embedding = combine(v["Categories"]).tolist()
        subcategory_embedding = combine(v["SubCategories"]).tolist()
        ship_mode_embedding = combine(v["Ship Mode Enc"]).tolist()

        returns_total = 0
        for v in v["Returns"]:
            if v == "Yes":
                returns_total+=1

        row = [k, total_sales_volume, total_discount, total_quantity, total_profit, postal_code, category_embedding, subcategory_embedding, returns_total, ship_mode_embedding, country, state, region, city]
        rows.append(row)

    pdf = pd.DataFrame(rows, columns=columns)
    return pdf
