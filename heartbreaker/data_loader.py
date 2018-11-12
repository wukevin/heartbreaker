"""
Code for loading in data into a cohesive data table.

The heart disease mortality data is listed by county, with no other identifying codes.
Therefore, the data will have be grouped by identifiers that consist of STATE,COUNTY
"""
import os
import sys
import re
import logging
import collections

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
assert os.path.isdir(DATA_DIR), "Cannot find data directory: {}".format(DATA_DIR)

HEART_DISEASE_FPATH = os.path.join(DATA_DIR, "CDC_heart_disease_mortality/Heart_Disease_Mortality_Data_Among_US_Adults_35_plus_by_State_Territory_and_County.csv")
assert os.path.isfile(HEART_DISEASE_FPATH)

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

def load_heart_disease_table(fname=HEART_DISEASE_FPATH, genders=["Overall"]):
    """
    Load in the heart disease table and return a DataFrame
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

def main():
    """Mostly for on the fly testing"""
    print(load_heart_disease_table())

if __name__ == "__main__":
    main()
