"""
Kai Wakura
CSE 163 Final Project: Is a UW Dorm Really That Expensive?

Tests the data cleaning and calculations in data_processing.py.
Uses small datasets with expected results.
"""

import pandas as pd

import data_processing as dp

TEST_DIR = 'test_data/'


def check_close(expected: float, actual: float) -> None:
    """
    Asserts that two numbers are equal, allowing a tiny difference for
    rounding error in decimal math.
    """
    assert abs(expected - actual) < 0.0001, \
        f'expected {expected}, got {actual}'


def test_clean_charges() -> None:
    """
    '.', blanks, negative codes and 0 should all become missing, and the
    input DataFrame should not be changed.
    """
    raw = pd.read_csv(TEST_DIR + 'test_cost_2024.csv',
                      na_values=dp.MISSING_CODES)
    cleaned = dp.clean_charges(raw)
    by_id = cleaned.set_index('UNITID')
    check_close(5000, by_id.loc[1, 'ROOMAMT'])
    assert pd.isna(by_id.loc[2, 'ROOMAMT'])
    assert pd.isna(by_id.loc[3, 'ROOMAMT'])
    assert pd.isna(by_id.loc[3, 'RMBRDAMT'])
    assert pd.isna(by_id.loc[4, 'BOARDAMT'])
    assert raw.loc[raw['UNITID'] == 3, 'ROOMAMT'].iloc[0] == -1


def test_add_combined_charge() -> None:
    """
    combined should be room + board when both exist, the combined-only
    charge otherwise, and NaN when neither is usable.
    """
    cost = dp.load_cost(TEST_DIR + 'test_cost_2024.csv').set_index('UNITID')
    check_close(8000, cost.loc[1, 'combined'])
    check_close(9000, cost.loc[2, 'combined'])
    check_close(18000, cost.loc[236948, 'combined'])
    assert pd.isna(cost.loc[3, 'combined'])
    assert pd.isna(cost.loc[4, 'combined'])
    assert not cost.loc[1, 'combined_only']
    assert cost.loc[2, 'combined_only']
    assert not cost.loc[3, 'combined_only']


def test_hud_by_county() -> None:
    """
    Checks the 10-digit to 5-digit county code conversion, the
    population-weighted mean for New England towns, the plain mean when
    every town has a population of 0, and the rent_pp conversion.
    """
    hud = pd.read_csv(TEST_DIR + 'test_hud.csv')
    counties = dp.hud_by_county(hud).set_index('COUNTYCD')
    assert sorted(counties.index) == [9001, 25001, 53033]
    check_close(2000, counties.loc[53033, 'fmr_2'])
    check_close(1750, counties.loc[9001, 'fmr_2'])
    check_close(1300, counties.loc[25001, 'fmr_2'])
    check_close(9000, counties.loc[53033, 'rent_pp'])
    assert counties.loc[9001, 'towns'] == 2


def test_school_year_cpi() -> None:
    """
    The average should include September through August only. The test
    file has a value of 999 just outside each window, which would change
    the answer if it were included.
    """
    cpi = dp.load_cpi(TEST_DIR + 'test_cpi.csv')
    check_close(105.5, dp.school_year_cpi(cpi, 2014))
    check_close(150, dp.school_year_cpi(cpi, 2024))


def test_build_q1() -> None:
    """
    Only public four-year schools with a price should be kept: the
    private school (3) and the community college (4) are dropped.
    """
    cost = dp.load_cost(TEST_DIR + 'test_cost_2024.csv')
    directory = dp.load_directory(TEST_DIR + 'test_hd.csv')
    q1 = dp.build_q1(cost, directory)
    assert sorted(q1['UNITID']) == [1, 2, 236948]
    assert q1['is_r1'].sum() == 2


def test_build_q2() -> None:
    """
    Checks nominal and real growth by hand with a CPI ratio of 1.5.
    School 1: 6000 -> 8000 is +33.3% nominal and -11.1% real.
    UW: 9000 -> 18000 is +100% nominal and +33.3% real.
    """
    cost_2014 = dp.load_cost(TEST_DIR + 'test_cost_2014.csv')
    cost_2024 = dp.load_cost(TEST_DIR + 'test_cost_2024.csv')
    directory = dp.load_directory(TEST_DIR + 'test_hd.csv')
    q2 = dp.build_q2(cost_2014, cost_2024, directory, 1.5)
    q2 = q2.set_index('UNITID')
    assert sorted(q2.index) == [1, 2, 236948]
    check_close(8000 / 6000 - 1, q2.loc[1, 'nominal_growth'])
    check_close((8000 / 6000) / 1.5 - 1, q2.loc[1, 'real_growth'])
    check_close(1.0, q2.loc[236948, 'nominal_growth'])
    check_close(2 / 1.5 - 1, q2.loc[236948, 'real_growth'])
    check_close(-0.1, q2.loc[2, 'nominal_growth'])


def test_build_q3() -> None:
    """
    School 2 has no room charge, so only school 1 and UW remain, both
    matched to King County rents.
    """
    cost = dp.load_cost(TEST_DIR + 'test_cost_2024.csv')
    directory = dp.load_directory(TEST_DIR + 'test_hd.csv')
    hud = pd.read_csv(TEST_DIR + 'test_hud.csv')
    q3 = dp.build_q3(cost, directory, dp.hud_by_county(hud))
    assert sorted(q3['UNITID']) == [1, 236948]
    assert (q3['rent_pp'] == 9000).all()


def test_rank_and_percentile() -> None:
    """
    Checks ranking with missing values and ties: rank 1 is the highest,
    and the percentile counts only strictly lower values.
    """
    df = pd.DataFrame({'UNITID': [1, 2, 3, 4, 236948],
                       'ROOMAMT': [5000, 13000, None, 12000, 12000]})
    rank, total, percentile = dp.rank_and_percentile(df, 'ROOMAMT')
    assert rank == 2
    assert total == 4
    check_close(25.0, percentile)
    missing = pd.DataFrame({'UNITID': [236948], 'ROOMAMT': [None]})
    assert dp.rank_and_percentile(missing, 'ROOMAMT') is None


def main() -> None:
    """
    Runs every test and prints a message when they all pass.
    """
    test_clean_charges()
    test_add_combined_charge()
    test_hud_by_county()
    test_school_year_cpi()
    test_build_q1()
    test_build_q2()
    test_build_q3()
    test_rank_and_percentile()
    print('All tests passed!')


if __name__ == '__main__':
    main()
