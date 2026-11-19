# Is a UW Dorm Really That Expensive?

CSE 163 final project by Kai Wakura. The project compares UW Seattle's dorm
prices with other U.S. public universities (rank), measures how fast they
grew compared with inflation from 2014-15 to 2024-25 (growth), and tests
whether local rent explains dorm prices (local rent).

## Files

- `data_processing.py`: loads, cleans and joins the five datasets
- `eda.py`: prints the exploratory data analysis results and saves the
  figures to `figures/`
- `test_project.py`: tests for `data_processing.py`, using the small files
  in `test_data/`
- `data/`: the five datasets (sources below)

## How to run

```
pip install pandas matplotlib seaborn scipy statsmodels openpyxl
python eda.py
python test_project.py
```

If reading the HUD Excel file fails with an `openpyxl` error, also run
`pip install python-calamine`; the code falls back to that reader
automatically.

## Data sources

| File | Source |
| --- | --- |
| `hd2024.csv`, `cost1_2024.csv` | IPEDS, https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx (year 2024) |
| `ic2014.csv` | IPEDS, same page (year 2014, Institutional Characteristics) |
| `FMR2024_final_revised.xlsx` | HUD, https://www.huduser.gov/portal/datasets/fmr.html (FY 2024 County Level Data) |
| `CPIAUCSL.csv` | FRED, https://fred.stlouisfed.org/series/CPIAUCSL |
