"""
Kai Wakura
CSE 163 Final Project: Is a UW Dorm Really That Expensive?

Exploratory data analysis. Running this file prints the size, missing
data and summary statistics of each raw and joined dataset, checks the
assumptions of the planned statistical tests, and saves every figure
in the figures/ folder.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

import data_processing as dp

FIGURE_DIR = 'figures/'
UW_COLOR = '#4b2e83'
OTHER_COLOR = '#9a9a9a'


def section(title: str) -> None:
    """
    Prints a section header so the output is easy to read.
    """
    print()
    print('=' * 70)
    print(title)
    print('=' * 70)


def describe_dataset(name: str, df: pd.DataFrame, quantitative: list[str],
                     categorical: list[str]) -> None:
    """
    Prints the shape, missing data, seven-number summary of each
    quantitative column and value counts of each categorical column.
    """
    section(name)
    print(f'Rows: {df.shape[0]}, columns: {df.shape[1]}')
    columns = quantitative + categorical
    missing = df[columns].isna().sum()
    percent = (missing / len(df) * 100).round(1)
    print('\nMissing values in variables of interest:')
    print(pd.DataFrame({'missing': missing, 'percent': percent}))
    if len(quantitative) > 0:
        print('\nSeven-number summaries:')
        summary = df[quantitative].describe().drop('count')
        print(summary.round(2))
    for column in categorical:
        print(f'\nValue counts for {column}:')
        print(df[column].value_counts(dropna=False))


def raw_overview(files: dict) -> None:
    """
    Prints the shape and percent missing of the key columns in each raw
    file, where files maps a name to (DataFrame, list of key columns).
    """
    section('RAW DATASETS')
    for name, (df, columns) in files.items():
        print(f'\n{name}: {df.shape[0]} rows, {df.shape[1]} columns')
        percent = (df[columns].isna().mean() * 100).round(1)
        print('Percent missing:', percent.to_dict())


def save(fig, filename: str) -> None:
    """
    Saves fig in the figures/ folder and closes it.
    """
    fig.tight_layout()
    fig.savefig(FIGURE_DIR + filename, dpi=200)
    plt.close(fig)


def plot_room_histogram(q1: pd.DataFrame) -> None:
    """
    Figure 1: histogram of 2024-25 room charges for all public four-year
    schools and for public R1 schools, with a line at UW.
    """
    uw = q1.loc[q1['UNITID'] == dp.UW_UNITID, 'ROOMAMT'].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = list(range(0, 21500, 500))
    ax.hist(q1['ROOMAMT'].dropna(), bins=bins, color=OTHER_COLOR,
            label='All public four-year')
    ax.hist(q1.loc[q1['is_r1'], 'ROOMAMT'].dropna(), bins=bins,
            color='#d4a017', label='Public R1')
    ax.axvline(uw, color=UW_COLOR, linewidth=2,
               label=f'UW Seattle (${uw:,.0f})')
    ax.set_title('2024-25 Dorm Room Charges at Public Four-Year Schools')
    ax.set_xlabel('Room charge, double occupancy (dollars per year)')
    ax.set_ylabel('Number of schools')
    ax.legend()
    save(fig, 'fig1_room_histogram.png')


def plot_top_r1(q1: pd.DataFrame) -> None:
    """
    Figure 2: bar chart of the 15 highest room charges among public R1
    schools, with UW highlighted.
    """
    r1 = q1[q1['is_r1']].dropna(subset=['ROOMAMT'])
    top = r1.sort_values('ROOMAMT').tail(15)
    colors = [UW_COLOR if unitid == dp.UW_UNITID else OTHER_COLOR
              for unitid in top['UNITID']]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top['INSTNM'], top['ROOMAMT'], color=colors)
    median = r1['ROOMAMT'].median()
    ax.axvline(median, color='black', linestyle='--',
               label=f'Public R1 median (${median:,.0f})')
    ax.set_title('Highest Room Charges, Public R1 Universities')
    ax.set_xlabel('Room charge, 2024-25 (dollars per year)')
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(loc='lower right')
    save(fig, 'fig2_top_r1.png')


def plot_cpi(cpi: pd.DataFrame) -> None:
    """
    Plots monthly CPI data and highlights the two school years used to
    calculate inflation.
    """
    years = cpi['observation_date'].str[:4].astype(int)
    months = cpi['observation_date'].str[5:7].astype(int)
    decimal_years = years + (months - 1) / 12
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(decimal_years, cpi['CPIAUCSL'], color='black')
    for start in [2014, 2024]:
        september = start + 8 / 12
        ax.axvspan(september, september + 1, color=UW_COLOR, alpha=0.2)
    ax.set_title('Consumer Price Index, 2014-2025 '
                 '(shaded: 2014-15 and 2024-25 school years)')
    ax.set_xlabel('Year')
    ax.set_ylabel('CPI (1982-84 = 100)')
    save(fig, 'fig3_cpi.png')


def plot_growth_scatter(q2: pd.DataFrame, cpi_ratio: float) -> None:
    """
    Figure 4: 2014-15 against 2024-25 combined charges for public
    four-year schools, with the line a school would follow if its price
    rose exactly with inflation.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    others = q2[q2['UNITID'] != dp.UW_UNITID]
    uw = q2[q2['UNITID'] == dp.UW_UNITID]
    ax.scatter(others['combined_2014'], others['combined_2024'],
               s=12, color=OTHER_COLOR, alpha=0.7, label='Other schools')
    ax.scatter(uw['combined_2014'], uw['combined_2024'], s=60,
               color=UW_COLOR, label='UW Seattle', zorder=3)
    old_prices = [0, q2['combined_2014'].max()]
    new_prices = [price * cpi_ratio for price in old_prices]
    ax.plot(old_prices, new_prices, color='black',
            linestyle='--',
            label=f'Grew with inflation (x {cpi_ratio:.2f})')
    ax.set_title('Room and Board, 2014-15 vs. 2024-25')
    ax.set_xlabel('Combined charge, 2014-15 (dollars)')
    ax.set_ylabel('Combined charge, 2024-25 (dollars)')
    ax.legend()
    save(fig, 'fig4_growth_scatter.png')


def plot_real_growth(q2: pd.DataFrame) -> None:
    """
    Figure 5: histogram of inflation-adjusted growth for public R1
    schools, with lines at 0 (no real growth) and at UW.
    """
    r1 = q2[q2['is_r1']]
    uw = r1.loc[r1['UNITID'] == dp.UW_UNITID, 'real_growth'].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(r1['real_growth'] * 100, bins=20, color=OTHER_COLOR)
    ax.axvline(0, color='black', linestyle='--',
               label='Same as inflation (0%)')
    ax.axvline(uw * 100, color=UW_COLOR, linewidth=2,
               label=f'UW Seattle (+{uw * 100:.0f}%)')
    ax.set_title('Inflation-Adjusted Room and Board Growth, 2014-15 to '
                 '2024-25 (Public R1)')
    ax.set_xlabel('Real growth (percent)')
    ax.set_ylabel('Number of schools')
    ax.legend()
    save(fig, 'fig5_real_growth.png')


def plot_rent_scatter(q3: pd.DataFrame, model) -> None:
    """
    Figure 6: local rent per student against dorm room charge, with the
    fitted regression line and the line where they are equal.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    others = q3[q3['UNITID'] != dp.UW_UNITID]
    uw = q3[q3['UNITID'] == dp.UW_UNITID]
    ax.scatter(others['rent_pp'], others['ROOMAMT'], s=12,
               color=OTHER_COLOR, alpha=0.7, label='Other schools')
    ax.scatter(uw['rent_pp'], uw['ROOMAMT'], s=60, color=UW_COLOR,
               label='UW Seattle', zorder=3)
    rents = [q3['rent_pp'].min(), q3['rent_pp'].max()]
    intercept, slope = model.params
    fitted = [intercept + slope * rent for rent in rents]
    ax.plot(rents, fitted, color='black',
            label='Fitted regression line')
    ax.plot(rents, rents, color='black', linestyle=':',
            label='Dorm = local rent')
    ax.set_title('Local Rent vs. Dorm Room Charge (Public Four-Year)')
    ax.set_xlabel('Half a 2-bedroom apartment for 9 months (dollars)')
    ax.set_ylabel('Room charge, 2024-25 (dollars)')
    ax.legend()
    save(fig, 'fig6_rent_scatter.png')


def plot_residuals(model) -> None:
    """
    Figure 7: residuals against fitted values (checks equal variance) and
    a normal Q-Q plot of the residuals (checks normality).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(model.fittedvalues, model.resid, s=10,
                    color=OTHER_COLOR)
    axes[0].axhline(0, color='black', linestyle='--')
    axes[0].set_title('Residuals vs. Fitted Values')
    axes[0].set_xlabel('Fitted room charge (dollars)')
    axes[0].set_ylabel('Residual (dollars)')
    stats.probplot(model.resid, dist='norm', plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot of Residuals')
    save(fig, 'fig7_residuals.png')


def q1_results(q1: pd.DataFrame) -> None:
    """
    Prints UW's rank and percentile for each charge, with and without
    schools that only report a combined charge (the sensitivity check).
    """
    section('Q1: UW RANKS (rank 1 = most expensive)')
    groups = {'Public four-year': q1, 'Public R1': q1[q1['is_r1']]}
    for name, group in groups.items():
        for column in ['ROOMAMT', 'BOARDAMT', 'combined']:
            rank, total, pct = dp.rank_and_percentile(group, column)
            print(f'{name:17} {column:9} rank {rank:3} of {total:3}, '
                  f'higher than {pct:.1f}%  median '
                  f'{group[column].median():,.0f}')
        separate = group[~group['combined_only']]
        rank, total, pct = dp.rank_and_percentile(separate, 'combined')
        print(f'{name:17} combined, separate reporters only: rank '
              f'{rank} of {total}, higher than {pct:.1f}%')


def q2_results(q2: pd.DataFrame) -> None:
    """
    Prints UW's growth, the R1 distribution, and the assumption checks
    and results for the Wilcoxon signed-rank test and the sign test.
    """
    section('Q2: GROWTH AND TEST ASSUMPTIONS')
    r1 = q2[q2['is_r1']]
    uw = r1[r1['UNITID'] == dp.UW_UNITID].iloc[0]
    print(f"UW: {uw['combined_2014']:,.0f} -> {uw['combined_2024']:,.0f},"
          f" nominal {uw['nominal_growth'] * 100:.1f}%,"
          f" real {uw['real_growth'] * 100:.1f}%")
    rank, total, pct = dp.rank_and_percentile(r1, 'real_growth')
    print(f'UW real growth rank {rank} of {total}, '
          f'higher than {pct:.1f}%')
    growth = r1['real_growth']
    above_zero = len(growth[growth > 0]) / len(growth) * 100
    print(f'R1 median real growth {growth.median() * 100:.1f}%'
          f', share above 0: {above_zero:.1f}%')
    print(f'Skewness of R1 real growth: {stats.skew(growth):.2f}')
    print(f'Mean {growth.mean() * 100:.1f}% vs median '
          f'{growth.median() * 100:.1f}%')
    wilcoxon = stats.wilcoxon(growth, alternative='greater')
    print(f'Wilcoxon signed-rank (H1: median > 0): '
          f'statistic {wilcoxon.statistic:.0f}, p = {wilcoxon.pvalue:.4g}')
    positives = len(growth[growth > 0])
    sign = stats.binomtest(positives, len(growth), 0.5,
                           alternative='greater')
    print(f'Sign test: {positives} of {len(growth)} positive, '
          f'p = {sign.pvalue:.4g}')


def q3_results(q3: pd.DataFrame):
    """
    Runs a regression comparing local rent and dorm prices. Prints the
    results and checks the model assumptions.
    """
    section('Q3: REGRESSION AND ASSUMPTION CHECKS')
    predictors = sm.add_constant(q3['rent_pp'])
    model = sm.OLS(q3['ROOMAMT'], predictors).fit()
    print(model.summary().tables[1])
    print(f'R-squared: {model.rsquared:.3f}, n = {int(model.nobs)}')
    print(f"Pearson r: {q3['rent_pp'].corr(q3['ROOMAMT']):.3f}")
    breusch_pagan = het_breuschpagan(model.resid, predictors)
    print(f'Breusch-Pagan test for unequal variance: '
          f'p = {breusch_pagan[1]:.4g}')
    jarque_bera = stats.jarque_bera(model.resid)
    print(f'Jarque-Bera test for normal residuals: '
          f'p = {jarque_bera.pvalue:.4g}')
    robust = model.get_robustcov_results(
        cov_type='cluster', groups=q3['COUNTYCD'])
    print(f'Slope with county-clustered SEs: {robust.params[1]:.3f}, '
          f'SE {robust.bse[1]:.3f}, p = {robust.pvalues[1]:.4g}')
    schools_per_county = q3.groupby('COUNTYCD')['UNITID'].count()
    shared = schools_per_county[schools_per_county > 1].sum()
    print(f'Schools sharing a county with another school: {shared}')
    studentized = model.get_influence().resid_studentized_external
    with_residuals = q3.copy()
    with_residuals['residual'] = model.resid
    with_residuals['studentized'] = studentized
    uw = with_residuals[with_residuals['UNITID'] == dp.UW_UNITID].iloc[0]
    print(f"UW: room {uw['ROOMAMT']:,.0f}, rent_pp {uw['rent_pp']:,.0f},"
          f" residual {uw['residual']:,.0f},"
          f" studentized {uw['studentized']:.2f}")
    rank, total, pct = dp.rank_and_percentile(with_residuals, 'residual')
    print(f'UW residual rank {rank} of {total}, higher than {pct:.1f}%')
    above_two = len(with_residuals[with_residuals['studentized'] > 2])
    print(f'Schools with studentized residual > 2: {above_two}')
    top = with_residuals.sort_values('residual', ascending=False).head(5)
    print(top[['INSTNM', 'ROOMAMT', 'rent_pp', 'residual']].round(0)
          .to_string())
    return model


def join_report(cost: pd.DataFrame, directory: pd.DataFrame,
                county_rents: pd.DataFrame) -> None:
    """
    Prints how many rows match in each join and which schools fail to
    match a HUD county.
    """
    section('JOIN MATCH RATES')
    merged = cost.merge(directory, left_on='UNITID', right_on='UNITID')
    matched = len(merged) / len(cost) * 100
    print(f'COST1_2024 rows matching HD2024 on UNITID: {matched:.1f}%')
    has_room = merged['ROOMAMT'].notna()
    candidates = merged[merged['public_4yr'] & has_room]
    with_rent = candidates.merge(county_rents, left_on='COUNTYCD',
                                 right_on='COUNTYCD', how='left')
    unmatched = with_rent[with_rent['fmr_2'].isna()]
    print(f'Public four-year schools with a room charge: '
          f'{len(candidates)}, not matched to HUD: {len(unmatched)}')
    print(unmatched[['INSTNM', 'STABBR', 'COUNTYCD']])
    print(f"HUD counties built from more than one town row: "
          f"{(county_rents['towns'] > 1).sum()}")


def main():
    """
    Loads and joins the datasets, prints the EDA results and saves the
    figures.
    """
    sns.set_theme(style='whitegrid')

    cost = dp.load_cost('data/cost1_2024.csv')
    cost_2014 = dp.load_cost('data/ic2014.csv')
    directory = dp.load_directory('data/hd2024.csv')
    hud = dp.load_hud('data/FMR2024_final_revised.xlsx')
    county_rents = dp.hud_by_county(hud)
    cpi = dp.load_cpi('data/CPIAUCSL.csv')

    raw_overview({
        'HD2024': (pd.read_csv('data/hd2024.csv', low_memory=False),
                   dp.HD_COLUMNS),
        'COST1_2024': (cost, dp.CHARGE_COLUMNS),
        'IC2014': (cost_2014, dp.CHARGE_COLUMNS),
        'HUD FY2024': (hud, ['fips', 'pop2020', 'fmr_2']),
        'CPI': (cpi, ['observation_date', 'CPIAUCSL']),
    })
    join_report(cost, directory, county_rents)

    cpi_2014 = dp.school_year_cpi(cpi, 2014)
    cpi_2024 = dp.school_year_cpi(cpi, 2024)
    cpi_ratio = cpi_2024 / cpi_2014
    section('CPI')
    print(f'2014-15 average {cpi_2014:.2f}, 2024-25 average '
          f'{cpi_2024:.2f}, ratio {cpi_ratio:.4f}')

    q1 = dp.build_q1(cost, directory)
    describe_dataset('Q1 DATASET: COST1_2024 + HD2024', q1,
                     ['ROOMAMT', 'BOARDAMT', 'combined'],
                     ['is_r1', 'combined_only'])
    q1_results(q1)
    plot_room_histogram(q1)
    plot_top_r1(q1)

    q2 = dp.build_q2(cost_2014, cost, directory, cpi_ratio)
    describe_dataset('Q2 DATASET: IC2014 + COST1_2024 + HD2024 + CPI', q2,
                     ['combined_2014', 'combined_2024', 'nominal_growth',
                      'real_growth'],
                     ['is_r1'])
    q2_results(q2)
    plot_cpi(cpi)
    plot_growth_scatter(q2, cpi_ratio)
    plot_real_growth(q2)

    q3 = dp.build_q3(cost, directory, county_rents)
    describe_dataset('Q3 DATASET: COST1_2024 + HD2024 + HUD', q3,
                     ['ROOMAMT', 'fmr_2', 'rent_pp'], ['is_r1'])
    model = q3_results(q3)
    plot_rent_scatter(q3, model)
    plot_residuals(model)


if __name__ == '__main__':
    main()
