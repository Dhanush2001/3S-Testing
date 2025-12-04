import numpy as np
from copy import deepcopy
import random
from sklearn.model_selection import train_test_split
import logging

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.impute import SimpleImputer

def load_seer_cutract_dataset(name="seer", seed=42):
    import sklearn

    def aggregate_grade(row):
        if row["grade_1.0"] == 1:
            return 1
        if row["grade_2.0"] == 1:
            return 2
        if row["grade_3.0"] == 1:
            return 3
        if row["grade_4.0"] == 1:
            return 4
        if row["grade_5.0"] == 1:
            return 5

    def aggregate_stage(row):
        if row["stage_1"] == 1:
            return 1
        if row["stage_2"] == 1:
            return 2
        if row["stage_3"] == 1:
            return 3
        if row["stage_4"] == 1:
            return 4
        if row["stage_5"] == 1:
            return 5

    def aggregate_treatment(row):
        if row["treatment_CM"] == 1:
            return 1
        if row["treatment_Primary hormone therapy"] == 1:
            return 2
        if row["treatment_Radical Therapy-RDx"] == 1:
            return 3
        if row["treatment_Radical therapy-Sx"] == 1:
            return 4

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
        "stage",
    ]

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment",
        "grade",
        "stage",
    ]

    # features = ['age', 'psa', 'comorbidities', 'treatment_CM', 'treatment_Primary hormone therapy',
    #         'treatment_Radical Therapy-RDx', 'treatment_Radical therapy-Sx', 'grade', 'stage']
    label = "mortCancer"
    try:
        df = pd.read_csv(f"../data/{name}.csv")
    except BaseException:
        df = pd.read_csv(f"../data/{name}.csv")

    df["grade"] = df.apply(aggregate_grade, axis=1)
    df["stage"] = df.apply(aggregate_stage, axis=1)
    df["treatment"] = df.apply(aggregate_treatment, axis=1)
    df["mortCancer"] = df["mortCancer"].astype(int)
    df["mort"] = df["mort"].astype(int)

    mask = df[label] == True
    df_dead = df[mask]
    df_survive = df[~mask]

    if name == "seer":
        n_samples = 10000
    else:
        n_samples = 1000
        
    df = pd.concat(
        [
            df_dead.sample(n_samples, random_state=seed),
            df_survive.sample(n_samples, random_state=seed),
        ]
    )
    df = sklearn.utils.shuffle(df, random_state=seed)
    df = df.reset_index(drop=True)
    return df[features], df[label]

def load_adult_data(split_size=0.3):
    """
    > This function loads the adult dataset, removes all the rows with missing values, and then splits the data into
    a training and test set
    Args:
      split_size: The proportion of the dataset to include in the test split.
    Returns:
      X_train, X_test, y_train, y_test, X, y
    """

    def process_dataset(df, random_state=42):
        """
        > This function takes a dataframe, maps the categorical variables to numerical values, and returns a
        dataframe with the numerical values
        Args:
          df: The dataframe to be processed
        Returns:
          a dataframe after the mapping
        """

        data = [df]

        salary_map = {" <=50K": 1, " >50K": 0}
        df["salary"] = df["salary"].map(salary_map).astype(int)

        df["sex"] = df["sex"].map({" Male": 1, " Female": 0}).astype(int)

        df["country"] = df["country"].replace(" ?", np.nan)
        df["workclass"] = df["workclass"].replace(" ?", np.nan)
        df["occupation"] = df["occupation"].replace(" ?", np.nan)

        df.dropna(how="any", inplace=True)

        for dataset in data:
            dataset.loc[
                dataset["country"] != " United-States",
                "country",
            ] = "Non-US"
            dataset.loc[
                dataset["country"] == " United-States",
                "country",
            ] = "US"

        df["country"] = df["country"].map({"US": 1, "Non-US": 0}).astype(int)

        df["marital-status"] = df["marital-status"].replace(
            [
                " Divorced",
                " Married-spouse-absent",
                " Never-married",
                " Separated",
                " Widowed",
            ],
            "Single",
        )
        df["marital-status"] = df["marital-status"].replace(
            [" Married-AF-spouse", " Married-civ-spouse"],
            "Couple",
        )

        df["marital-status"] = df["marital-status"].map(
            {"Couple": 0, "Single": 1},
        )

        rel_map = {
            " Unmarried": 0,
            " Wife": 1,
            " Husband": 2,
            " Not-in-family": 3,
            " Own-child": 4,
            " Other-relative": 5,
        }

        df["relationship"] = df["relationship"].map(rel_map)

        race_map = {
            " White": 0,
            " Amer-Indian-Eskimo": 1,
            " Asian-Pac-Islander": 2,
            " Black": 3,
            " Other": 4,
        }

        df["race"] = df["race"].map(race_map)

        def f(x):
            if (
                x["workclass"] == " Federal-gov"
                or x["workclass"] == " Local-gov"
                or x["workclass"] == " State-gov"
            ):
                return "govt"
            elif x["workclass"] == " Private":
                return "private"
            elif (
                x["workclass"] == " Self-emp-inc"
                or x["workclass"] == " Self-emp-not-inc"
            ):
                return "self_employed"
            else:
                return "without_pay"

        df["employment_type"] = df.apply(f, axis=1)

        employment_map = {
            "govt": 0,
            "private": 1,
            "self_employed": 2,
            "without_pay": 3,
        }

        df["employment_type"] = df["employment_type"].map(employment_map)
        df.drop(
            labels=[
                "workclass",
                "education",
                "occupation",
            ],
            axis=1,
            inplace=True,
        )
        df.loc[(df["capital-gain"] > 0), "capital-gain"] = 1
        df.loc[(df["capital-gain"] == 0, "capital-gain")] = 0

        df.loc[(df["capital-loss"] > 0), "capital-loss"] = 1
        df.loc[(df["capital-loss"] == 0, "capital-loss")] = 0

        return df

    try:
        df = pd.read_csv("data/adult.csv", delimiter=",")
    except BaseException:
        df = pd.read_csv("../data/adult.csv", delimiter=",")

    df = process_dataset(df)

    df_sex_1 = df.query("sex ==1")

    salary_1_idx = df.query("sex == 0 & salary == 1")
    salary_0_idx = df.query("sex == 0 & salary == 0")

    X = df.drop(["salary"], axis=1)
    X = X.drop('fnlwgt', axis=1)
    y = df["salary"]

    # Creation of Train and Test dataset
    random.seed(a=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_size,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test, X, y


def load_covid_dataset(seed=42, drop_SG_UF_NOT=True):


    # Read your covid.csv
    df_ALL = pd.read_csv("../data/covid.csv")

    # Target is the 'death' column
    y = df_ALL["death"].astype(int)

    # Start from all feature columns (drop the label)
    df = df_ALL.drop(columns=["death"]).copy()

    # Race dummy columns in your file
    race_cols = ["Branca", "Preta", "Amarela", "Parda", "Indigena"]

    # Create a single 'Race' categorical column from the one-hot columns
    df["Race"] = df[race_cols].idxmax(axis=1)

    # Create a synthetic 'SG_UF_NOT' column so the notebook can split
    # north vs south. We just assign half the rows to 'SP' and half to 'AM'
    # in a reproducible way using the given seed.
    rng = np.random.default_rng(seed)
    mask = rng.random(df.shape[0]) < 0.5
    df["SG_UF_NOT"] = np.where(mask, "SP", "AM")

    # Match the original API: optionally drop SG_UF_NOT, always drop Race
    if drop_SG_UF_NOT:
        df = df.drop(columns=["Race", "SG_UF_NOT"])
    else:
        df = df.drop(columns=["Race"])

    # Attach label as 'y'
    df["y"] = y

    # Return (X, y, df) as in the original function
    return df.drop(columns=["y"]), df["y"], df



# def load_drug_dataset():
#     from sklearn.impute import SimpleImputer
#     data = pd.read_csv('../data/Drug_Consumption.csv')

#     #Drop overclaimers, Semer, and other nondrug columns
#     data = data.drop(data[data['Semer'] != 'CL0'].index)
#     data = data.drop(['Semer', 'Caff', 'Choc'], axis=1)
#     data.reset_index()

#     # Binary encode gender
#     data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)

#     # Encode ordinal features
#      # Encode ordinal features
#     ordinal_features = [
#         'Age',
#         'Education',
#         'Alcohol',
#         'Amyl',
#         'Amphet',
#         'Benzos',
#         'Cannabis',
#         'Coke',
#         'Crack',
#         'Ecstasy',
#         'Heroin',
#         'Ketamine',
#         'Legalh',
#         'LSD',
#         'Meth',
#         'Mushrooms',
#         'Nicotine',
#         'VSA',
#     ]

#     # Build ordinal orderings dynamically from the actual data
#     # This works for both "18-24" style labels and "Ag1"/"Ag2"/"Ag3" style labels.
#     ordinal_orderings = []
#     for col in ordinal_features:
#         # Get unique values, sort them, and use that as the ordering
#         vals = sorted(data[col].unique())
#         ordinal_orderings.append(list(vals))

#     # Ordinal encoding
#     def ordinal_encoder(df, columns, ordering):
#         df = df.copy()
#         for column, order in zip(columns, ordering):
#             df[column] = df[column].apply(lambda x: order.index(x))
#         return df

#     # Nominal features
#     nominal_features = ['Country', 'Ethnicity']

#     # Convert nominal features to category codes
#     def cat_converter(df, columns):
#         df = df.copy()
#         for column in columns:
#             df[column] = df[column].astype('category').cat.codes
#         return df

#     # Apply encodings
#     data = ordinal_encoder(data, ordinal_features, ordinal_orderings)
#     data = cat_converter(data, nominal_features)

#     nic_df = data.copy()
#     nic_df['y'] = nic_df['Nicotine'].apply(lambda x: 1 if x not in [0,1] else 0)
#     nic_df = nic_df.drop(['ID','Nicotine'], axis=1)

#     return nic_df.drop(columns=['y']), nic_df['y'], nic_df


# def load_bank_dataset(seed=0):
#     import pandas as pd

#     df = pd.read_csv('../data/Base.csv')
#     for col in ["payment_type", "employment_status", "housing_status", "source", "device_os"]:
#         df[col] = df[col].astype("category").cat.codes

#     df.rename(columns={'fraud_bool': 'y'}, inplace=True)

#     mask = df['y'] == True
#     df_fraud = df[mask]
#     df_no = df[~mask]

#     n_samples = 5000
#     df = pd.concat(
#         [
#             df_fraud.sample(n_samples, random_state=seed),
#             df_no.sample(n_samples, random_state=seed),
#         ]
#     )
#     from sklearn.utils import shuffle
#     df = shuffle(df, random_state=seed)

#     return df.drop(columns=['y']), df['y'], df

"""def load_drug_dataset():
    from sklearn.impute import SimpleImputer
    data = pd.read_csv('../data/Drug_Consumption.csv')

    #Drop overclaimers, Semer, and other nondrug columns
    data = data.drop(data[data['Semer'] != 'CL0'].index)
    data = data.drop(['Semer', 'Caff', 'Choc'], axis=1)
    data.reset_index()

    # Binary encode gender
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)

    # Encode ordinal features
    ordinal_features = ['Age', 
                        'Education',
                        'Alcohol',
                        'Amyl',
                        'Amphet',
                        'Benzos',
                        'Cannabis',
                        'Coke',
                        'Crack',
                        'Ecstasy',
                        'Heroin',
                        'Ketamine',
                        'Legalh',
                        'LSD',
                        'Meth',
                        'Mushrooms',
                        'Nicotine',
                        'VSA'    ]

    # Define ordinal orderings
    ordinal_orderings = [
        ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        ['Left school before 16 years', 
        'Left school at 16 years', 
        'Left school at 17 years', 
        'Left school at 18 years',
        'Some college or university, no certificate or degree',
        'Professional certificate/ diploma',
        'University degree',
        'Masters degree',
        'Doctorate degree'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6']
    ]

    # Nominal features
    nominal_features = ['Country',
                        'Ethnicity']

    #Create function for ordinal encoding
    def ordinal_encoder(df, columns, ordering):
        df = df.copy()
        for column, ordering in zip(ordinal_features, ordinal_orderings):
            df[column] = df[column].apply(lambda x: ordering.index(x))
        return df

    def cat_converter(df, columns):
        df = df.copy()
        for column in columns:
            df[column] = df[column].astype('category').cat.codes
        return df

    data = ordinal_encoder(data, ordinal_features, ordinal_orderings)
    data = cat_converter(data, nominal_features)

    nic_df = data.copy()
    nic_df['y'] = nic_df['Nicotine'].apply(lambda x: 1 if x not in [0,1] else 0)
    nic_df = nic_df.drop(['ID','Nicotine'], axis=1)

    return nic_df.drop(columns=['y']), nic_df['y'], nic_df"""

"""
def load_drug_dataset(seed: int = 0, csv_path: str = "../data/Drug_Consumption.csv"):

    # Read raw CSV
    data = pd.read_csv(csv_path)

    # 1. Filter out "over-claimers" and drop unused columns
    data = data[data["semer"] == "CL0"].copy()

    drop_cols = ["semer", "caff", "choc"]
    existing_drop = [c for c in drop_cols if c in data.columns]
    data = data.drop(columns=existing_drop)

    # 2. Convert CL0–CL6 codes to integers
    cl_cols = [
        "alcohol",
        "amphet",
        "amyl",
        "benzos",
        "cannabis",
        "coke",
        "crack",
        "ecstasy",
        "heroin",
        "ketamine",
        "legalh",
        "lsd",
        "meth",
        "mushrooms",
        "nicotine",
        "vsa",
    ]
    cl_map = {f"CL{i}": i for i in range(7)}

    for col in cl_cols:
        if col in data.columns:
            data[col] = data[col].map(cl_map).astype("int8")

    # 3. Build binary label from nicotine usage
    if "nicotine" not in data.columns:
        raise KeyError("Expected `nicotine` column in Drug_Consumption.csv")

    # CL0/CL1 -> 0, CL2+ -> 1
    data["y"] = (data["nicotine"] >= 2).astype(int)

    # 4. Drop identifier and nicotine (target feature) from features
    for col in ["id", "nicotine"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    # 5. Rename columns to match the notebook
    rename_map = {
        "age": "Age",
        "gender": "Gender",
        "education": "Education",
        "country": "Country",
        "ethnicity": "Ethnicity",
        "impulsive": "Impulsive",
        "ss": "SS",
        "alcohol": "Alcohol",
        "amphet": "Amphet",
        "amyl": "Amyl",
        "benzos": "Benzos",
        "cannabis": "Cannabis",
        "coke": "Coke",
        "crack": "Crack",
        "ecstasy": "Ecstasy",
        "heroin": "Heroin",
        "ketamine": "Ketamine",
        "legalh": "Legalh",
        "lsd": "LSD",
        "meth": "Meth",
        "mushrooms": "Mushrooms",
        "vsa": "VSA",
    }
    data = data.rename(columns=rename_map)

    # 6. Split X, y and return
    X = data.drop(columns=["y"])
    y = data["y"]

    return X, y, data
    """
import pandas as pd

def load_drug_dataset(
    seed: int = 0,
    csv_path: str = "../data/Drug_Consumption.csv",
):
    """
    Load and preprocess the Drug_Consumption dataset.

    This version is a cleaned-up, more robust version of the original function,
    written in the same style as your newer implementation.

    Steps
    -----
    1. Load the CSV.
    2. Remove "over-claimers" (semer != CL0) and drop unused columns.
    3. Convert CL0–CL6 usage codes to integers 0–6 for selected drug columns.
    4. Build a binary label `y` from nicotine usage:
         - 0 = CL0/CL1 (non/very light user)
         - 1 = CL2–CL6 (medium/heavy user)
    5. Drop identifier and nicotine columns from the feature matrix.
    6. Optionally rename columns to match notebook conventions (TitleCase).

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix (no label column).
    y : pandas.Series
        Binary label: 1 = medium/heavy nicotine user, 0 = non/very light user.
    data : pandas.DataFrame
        Same as X but with the label column `y` included.
    """
    # 0. Load raw CSV
    data = pd.read_csv(csv_path)

    # Ensure we're working with lowercase column names internally
    # (this matches the Kaggle Drug Consumption dataset style)
    data.columns = [c.lower() for c in data.columns]

    # 1. Filter out "over-claimers" and drop unused columns
    if "semer" in data.columns:
        data = data[data["semer"] == "CL0"].copy()

        drop_cols = ["semer", "caff", "choc"]
        existing_drop = [c for c in drop_cols if c in data.columns]
        if existing_drop:
            data = data.drop(columns=existing_drop)

    # 2. Convert CL0–CL6 codes to integers for drug usage columns
    cl_cols = [
        "alcohol",
        "amyl",
        "amphet",
        "benzos",
        "cannabis",
        "coke",
        "crack",
        "ecstasy",
        "heroin",
        "ketamine",
        "legalh",
        "lsd",
        "meth",
        "mushrooms",
        "nicotine",
        "vsa",
    ]
    cl_map = {f"CL{i}": i for i in range(7)}

    for col in cl_cols:
        if col in data.columns:
            data[col] = data[col].map(cl_map).astype("int8")

    # 3. Build binary label from nicotine usage
    if "nicotine" not in data.columns:
        raise KeyError("Expected `nicotine` column in Drug_Consumption.csv after loading")

    # CL0/CL1 -> 0, CL2+ -> 1
    data["y"] = (data["nicotine"] >= 2).astype(int)

    # 4. Drop identifier and nicotine (target feature) from X
    for col in ["id", "nicotine"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    # 5. Rename columns to match the notebook (TitleCase etc.)
    rename_map = {
        "age": "Age",
        "gender": "Gender",
        "education": "Education",
        "country": "Country",
        "ethnicity": "Ethnicity",
        "impulsive": "Impulsive",
        "ss": "SS",
        "alcohol": "Alcohol",
        "amphet": "Amphet",
        "amyl": "Amyl",
        "benzos": "Benzos",
        "cannabis": "Cannabis",
        "coke": "Coke",
        "crack": "Crack",
        "ecstasy": "Ecstasy",
        "heroin": "Heroin",
        "ketamine": "Ketamine",
        "legalh": "Legalh",
        "lsd": "LSD",
        "meth": "Meth",
        "mushrooms": "Mushrooms",
        "vsa": "VSA",
    }
    # Only rename columns that actually exist
    active_rename_map = {k: v for k, v in rename_map.items() if k in data.columns}
    if active_rename_map:
        data = data.rename(columns=active_rename_map)

    # 6. Split X, y and return
    X = data.drop(columns=["y"])
    y = data["y"]

    return X, y, data


def load_support_dataset(seed=42):
    """
    Load and preprocess the SUPPORT dataset.

    - Finds support_data.csv in a few reasonable locations.
    - Encodes income -> salary (0..3).
    - Encodes race -> 0..3 for {white, black, asian, hispanic}.
    - Encodes sex: male=1, female=0.
    - Drops non-numeric text columns not needed for modelling.
    - Imputes missing numeric values with median (SimpleImputer).
    - Returns: X, y, Data ; where Data = X with 'y' column appended.
    """
    import os
    from sklearn.impute import SimpleImputer

    # 1) Robust path handling
    candidates = [
        "../data/support_data.csv",   # original expected location
        "data/support_data.csv",      # common alternative
        "support_data.csv",           # same folder as notebook / script
    ]

    csv_path = None
    for p in candidates:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(
            "support_data.csv not found in ../data, data/, or current directory."
        )

    df = pd.read_csv(csv_path)

    # 2) Encode income -> salary 0..3 (drop unknown income rows)
    income_order = ["under $11k", "$11-$25k", "$25-$50k", ">$50k"]
    df = df[df["income"].isin(income_order)].copy()
    income_map = {v: i for i, v in enumerate(income_order)}
    df["salary"] = df["income"].map(income_map)

    # 3) Encode race -> 0..3 (drop 'other' / NaN)
    race_order = ["white", "black", "asian", "hispanic"]
    df = df[df["race"].isin(race_order)].copy()
    race_map = {v: i for i, v in enumerate(race_order)}
    df["race"] = df["race"].map(race_map)

    # 4) Binary-encode sex
    df["sex"] = df["sex"].map({"male": 1, "female": 0})

    # 5) Drop non-numeric / text columns that would break sklearn
    drop_cols = ["income", "d.time", "dzgroup", "dzclass", "ca", "dnr", "sfdm2"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # 6) Rename label column
    df.rename(columns={"death": "y"}, inplace=True)

    # 7) Split into X / y
    X = df.drop(columns=["y"])
    y = df["y"].astype(int)

    # 8) Impute missing values with median so sklearn models don’t crash
    num_cols = X.columns
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # 9) Rebuild Data = X + y for the notebook’s convenience
    Data = X.copy()
    Data["y"] = y

    return X, y, Data



def load_bank_dataset(seed=0):
    import pandas as pd

    df = pd.read_csv('../data/Base.csv')
    for col in ["payment_type", "employment_status", "housing_status", "source", "device_os"]:
        df[col] = df[col].astype("category").cat.codes

    df.rename(columns={'fraud_bool': 'y'}, inplace=True)

    mask = df['y'] == True
    df_fraud = df[mask]
    df_no = df[~mask]

    n_samples = 5000
    df = pd.concat(
        [
            df_fraud.sample(n_samples, random_state=seed),
            df_no.sample(n_samples, random_state=seed),
        ]
    )
    from sklearn.utils import shuffle
    df = shuffle(df, random_state=seed)

    return df.drop(columns=['y']), df['y'], df