"""
Kai Wakura
CSE 163 Final Project: Is a UW Dorm Really That Expensive?

Functions that load, clean and join the five datasets used in the
project: IPEDS HD2024, IPEDS COST1_2024, IPEDS IC2014, HUD Fair Market
Rents (FY 2024, county level) and the FRED CPIAUCSL series.
"""

import pandas as pd

UW_UNITID = 236948
R1_CODE = 15
CHARGE_COLUMNS = ['ROOMAMT', 'BOARDAMT', 'RMBRDAMT']
HD_COLUMNS = ['UNITID', 'INSTNM', 'STABBR', 'CONTROL', 'ICLEVEL',
              'C21BASIC', 'COUNTYCD']
MISSING_CODES = ['.', ' ']
MONTHS_IN_SCHOOL_YEAR = 9
ROOMMATES = 2


def clean_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df where every charge that is 0 or negative is
    missing. IPEDS uses negative numbers as codes for "not applicable",
    and no school offers free housing, so 0 is treated as missing too.
    """
    result = df.copy()
    for column in CHARGE_COLUMNS:
        not_a_price = result[column] <= 0
        result.loc[not_a_price, column] = None
    return result


def add_combined_charge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the combined room and board charge. Uses RMBRDAMT when
    separate charges are not available. Also marks schools that only
    report a combined charge.
    """
    result = df.copy()
    result['combined'] = result['ROOMAMT'] + result['BOARDAMT']
    separate_missing = result['combined'].isna()
    result.loc[separate_missing, 'combined'] = result['RMBRDAMT']
    has_combined = result['RMBRDAMT'].notna()
    result['combined_only'] = separate_missing & has_combined
    return result


def load_cost(path: str) -> pd.DataFrame:
    """
    Loads an IPEDS charges file (COST1_2024 or IC2014), cleans the charge
    columns and adds the combined charge. IPEDS stores some missing values
    as '.', so those are read as missing. Returns UNITID plus the charge
    columns.
    """
    df = pd.read_csv(path, na_values=MISSING_CODES, low_memory=False)
    df = clean_charges(df)
    df = add_combined_charge(df)
    return df[['UNITID'] + CHARGE_COLUMNS + ['combined', 'combined_only']]


def load_directory(path: str) -> pd.DataFrame:
    """
    Loads IPEDS HD2024 and returns the columns used in this project, plus
    True/False columns 'public_4yr' and 'is_r1'.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[HD_COLUMNS].copy()
    is_public = df['CONTROL'] == 1
    is_four_year = df['ICLEVEL'] == 1
    df['public_4yr'] = is_public & is_four_year
    df['is_r1'] = df['C21BASIC'] == R1_CODE
    return df


def load_hud(path: str) -> pd.DataFrame:
    """
    Reads the HUD Fair Market Rent Excel file. Some versions of openpyxl
    can't read this file, so if the default reader fails we use the
    calamine reader instead (pip install python-calamine).
    """
    try:
        return pd.read_excel(path)
    except ValueError:
        return pd.read_excel(path, engine='calamine')


def hud_by_county(hud: pd.DataFrame) -> pd.DataFrame:
    """
    Groups HUD rental data by county by dropping the last five digits of
    the FIPS code. Uses population-weighted rent for counties with
    multiple towns and estimates rent per student for 9 months.
    """
    df = hud[['fips', 'pop2020', 'fmr_2']].copy()
    df['COUNTYCD'] = df['fips'] // 100000
    df['weighted'] = df['fmr_2'] * df['pop2020']

    by_county = df.groupby('COUNTYCD')
    counties = pd.DataFrame()
    counties['weighted'] = by_county['weighted'].sum()
    counties['pop'] = by_county['pop2020'].sum()
    counties['plain_mean'] = by_county['fmr_2'].mean()
    counties['towns'] = by_county['fips'].count()

    counties['fmr_2'] = counties['plain_mean']
    has_pop = counties['pop'] > 0
    counties.loc[has_pop, 'fmr_2'] = (counties.loc[has_pop, 'weighted']
                                      / counties.loc[has_pop, 'pop'])
    counties['rent_pp'] = (counties['fmr_2'] / ROOMMATES
                           * MONTHS_IN_SCHOOL_YEAR)
    return counties[['fmr_2', 'rent_pp', 'towns']].reset_index()


def load_cpi(path: str) -> pd.DataFrame:
    """
    Loads the FRED CPIAUCSL CSV. Dates stay as 'YYYY-MM-DD' strings, which
    compare in date order.
    """
    return pd.read_csv(path)


def school_year_cpi(cpi: pd.DataFrame, start_year: int) -> float:
    """
    Returns the average CPIAUCSL from September of start_year through
    August of start_year + 1. For example, start_year=2014 gives the
    2014-15 school year.
    """
    start = str(start_year) + '-09-01'
    end = str(start_year + 1) + '-08-01'
    after_start = cpi['observation_date'] >= start
    before_end = cpi['observation_date'] <= end
    return cpi.loc[after_start & before_end, 'CPIAUCSL'].mean()


def build_q1(cost: pd.DataFrame, directory: pd.DataFrame) -> pd.DataFrame:
    """
    Question 1 dataset: 2024-25 charges joined to the directory on
    UNITID, keeping public four-year schools that report a room charge
    or a combined charge.
    """
    merged = cost.merge(directory, left_on='UNITID', right_on='UNITID')
    has_price = merged['ROOMAMT'].notna() | merged['combined'].notna()
    result = merged[merged['public_4yr'] & has_price]
    return result.reset_index(drop=True)


def build_q2(cost_2014: pd.DataFrame, cost_2024: pd.DataFrame,
             directory: pd.DataFrame, cpi_ratio: float) -> pd.DataFrame:
    """
    Combines housing prices from 2014 and 2024 for public four-year
    schools. Calculates both nominal and inflation-adjusted growth.
    """
    old = cost_2014[['UNITID', 'combined']].rename(
        columns={'combined': 'combined_2014'})
    new = cost_2024[['UNITID', 'combined']].rename(
        columns={'combined': 'combined_2024'})
    merged = old.merge(new, left_on='UNITID', right_on='UNITID')
    merged = merged.merge(directory, left_on='UNITID', right_on='UNITID')

    has_2014 = merged['combined_2014'].notna()
    has_2024 = merged['combined_2024'].notna()
    result = merged[merged['public_4yr'] & has_2014 & has_2024].copy()
    result['nominal_growth'] = (result['combined_2024']
                                / result['combined_2014'] - 1)
    result['real_growth'] = ((1 + result['nominal_growth'])
                             / cpi_ratio - 1)
    return result.reset_index(drop=True)


def build_q3(cost: pd.DataFrame, directory: pd.DataFrame,
             county_rents: pd.DataFrame) -> pd.DataFrame:
    """
    Question 3 dataset: 2024-25 charges joined to the directory on
    UNITID, then to county rents on COUNTYCD. Keeps public four-year
    schools with a room charge and a matching county.
    """
    merged = cost.merge(directory, left_on='UNITID', right_on='UNITID')
    merged = merged.merge(county_rents, left_on='COUNTYCD',
                          right_on='COUNTYCD')
    has_room = merged['ROOMAMT'].notna()
    result = merged[merged['public_4yr'] & has_room]
    return result.reset_index(drop=True)


def rank_and_percentile(df: pd.DataFrame, column: str,
                        unitid: int = UW_UNITID):
    """
    Returns (rank, total, percentile) for the school with this unitid,
    where rank 1 is the highest value of column among rows with a value,
    total is the number of those rows, and percentile is the percent of
    them with a lower value. Returns None if the school has no value.
    """
    values = df[['UNITID', column]].dropna()
    school = values[values['UNITID'] == unitid]
    if len(school) == 0:
        return None
    value = school[column].iloc[0]
    rank = len(values[values[column] > value]) + 1
    percentile = len(values[values[column] < value]) / len(values) * 100
    return rank, len(values), percentile
