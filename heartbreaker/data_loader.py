"""
Code for loading in data into a cohesive data table.

The heart disease mortality data is listed by county, with no other identifying codes.
Therefore, the data will have be grouped by identifiers that consist of STATE,COUNTY
"""
import os
import sys
import logging
import glob
import collections

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
assert os.path.isdir(DATA_DIR), "Cannot find data directory: {}".format(DATA_DIR)

HEART_DISEASE_FPATH = os.path.join(DATA_DIR, "CDC_heart_disease_mortality/Heart_Disease_Mortality_Data_Among_US_Adults_35_plus_by_State_Territory_and_County.csv")
assert os.path.isfile(HEART_DISEASE_FPATH)
USDA_FOOD_ATLAS_DIR = os.path.join(DATA_DIR, "USDA_food_environment_atlas")
assert os.path.isdir(USDA_FOOD_ATLAS_DIR)

def determine_most_prevalent(x):
    """Given an iterable, find the most common element"""
    counter = collections.Counter(x)
    return counter.most_common(1)[0][0]  # most_common returns a list of tuples of (element, count); we don't care about count

def homogenize_county_name(county):
    """Homogenize county name"""
    assert county and isinstance(county, str)
    county = county.strip()  # Strip leading/trailing whitespace
    if county.endswith(" County"):  # Strip trailing " County" from name
        county = county.replace(" County", "")
    return "_".join(county.split()).lower()  # Replace whitespace with "_" and return lowercase

def homogenize_state_abbrev(state_abbrev):
    """Homogenize state name"""
    assert state_abbrev and isinstance(state_abbrev, str)
    assert len(state_abbrev) == 2
    return state_abbrev.upper()  # Return uppercase

def load_heart_disease_table(fname=HEART_DISEASE_FPATH):
    """
    Load in the heart disease table and return a DataFrame with index values of STATE|county
    For now, we are discarding all data that is:
    - Stratified by race or by gender
    - Marked as "Insufficient Data"
    - Not of the most common type of measurement
    """
    # Read in the csv file into a data frame
    df = pd.read_csv(fname, engine='c', low_memory=False)
    # Remove data that is broken down by gender or by race.
    df.drop(index=[i for i, row in df.iterrows() if row['Stratification1'] != "Overall" or row['Stratification2'] != "Overall"], inplace=True)
    # Remove data that is marked as insufficient data
    df.drop(index=[i for i, row in df.iterrows() if row['Data_Value_Footnote'] == "Insufficient Data"], inplace=True)
    # Some sanity checks to make sure that our data is uniformly measuring the same thing
    assert len(set(df['Topic'])) == 1
    assert len(set(df['Data_Value_Unit'])) == 1
    if not len(set(df['Data_Value_Type'])) == 1:
        majority = determine_most_prevalent(df['Data_Value_Type'])
        df.drop(index=[index for index, row in df.iterrows() if row['Data_Value_Type'] != majority], inplace=True)
    assert len(set(df['Data_Value_Type'])) == 1

    # Build a new dataframe containing only information that we want.
    county_identifiers = []
    county_values = []
    for _i, row in df.iterrows():
        county = homogenize_county_name(row['LocationDesc'])
        state = homogenize_state_abbrev(row['LocationAbbr'])
        county_identifiers.append("|".join([state, county]))
        county_values.append(float(row['Data_Value']))

    retval = pd.DataFrame(
        data=county_values,
        index=county_identifiers,
        columns=['heart_disease_mortality'],
        dtype=float,
    )
    return retval

def load_usda_food_env_table(fname):
    """
    General function for reading in any of the csv files that come from the USDA food
    environment atlas (excluding the supplementary tables). In doing so, it drops all
    non-numeric data. As a data cleaning measure, we also drop all instances of any county
    that shows up more than once.
    """
    if os.path.basename(fname).startswith("supplemental"):
        raise NotImplementedError("Cannot read supplemental tables")
    df = pd.read_csv(fname, engine='c', low_memory=False)

    # Reindex according to our unified county naming scheme
    homogenized_identifiers = ["|".join([homogenize_state_abbrev(row['State']), homogenize_county_name(row['County'])]) for _i, row in df.iterrows()]
    assert len(homogenized_identifiers) == df.shape[0]
    df.index = homogenized_identifiers
    
    # Find duplicated rows and drop them
    dup_counter = collections.Counter(homogenized_identifiers)
    duplicated = [identifier for identifier, count in dup_counter.most_common() if count > 1]
    df.drop(index=duplicated, inplace=True)
    assert all([dup not in df.index for dup in duplicated])  # Sanity check
    
    # Drop any data that is not numeric (including old state/county labels)
    df.drop(columns=df.select_dtypes(exclude='number'), inplace=True)
    if "FIPS" in df.columns:  # Needs special handling because this will appear numeric
        df.drop(columns='FIPS', inplace=True)
        
    # Drop any data that is observed AFTER our heart disease data (2014)
    future_knowledge_cols = []
    for column in df.columns:
        try:
            year = int(column[-2:])
            if (year > 14):
                future_knowledge_cols.append(column)
        except:
            continue

    df.drop(columns=future_knowledge_cols, inplace=True)

    return df

def load_all_data(heart_disease_fname=HEART_DISEASE_FPATH, usda_food_env_folder=USDA_FOOD_ATLAS_DIR):
    """
    Loads in all the data and joins them, returning a pandas dataframe where each row is a county
    and columns represent measurements of a certain feature.
    """
    # Everything is inner joined starting from here
    logging.info("Reading in {}".format(heart_disease_fname))
    heart_disease_df = load_heart_disease_table(heart_disease_fname)

    # Read in the food env data and perform inner joins on our unified county identifier
    for match in glob.glob(os.path.join(usda_food_env_folder, "*.csv")):  # Query for all the csv files
        if os.path.basename(match).startswith("supplemental") or os.path.basename(match) == "variable_list.csv":
            continue  # Skip certain files
        logging.info("Reading in {}".format(match))
        df = load_usda_food_env_table(match)
        # Update the heart disease dataframe with the result of the inner join
        heart_disease_df = pd.merge(heart_disease_df, df, 'inner', left_index=True, right_index=True)

    return heart_disease_df

def main():
    """Mostly for on the fly testing"""
    print(load_all_data())

if __name__ == "__main__":
    main()
