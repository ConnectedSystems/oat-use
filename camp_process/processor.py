import numpy as np
import pandas as pd

def collate_columns(data, column, reset_index=True):
    """Collate specified column from different DataFrames

    :param data: dict, of pd.DataFrames spread by {ZoneID: {ScenarioID: result_dataframe}}
                TODO: {ZoneID: {Climate Scenario: {ScenarioID: result_dataframe}}}
    :param column: str, name of column to collate into single dataframe

    :returns: dict, of pd.DataFrame of column data collated for each Zone
    """
    collated = {}
    for zone, res in list(data.items()):
        # need pd.Series to allow columns of different lengths
        zone_d = {scen: pd.Series(res[scen].loc[:, column].values) for scen in res}
        collated[zone] = pd.DataFrame(zone_d)
        if reset_index:
            collated[zone].reset_index(drop=True, inplace=True)
    # End for

    return collated
# End collate_columns()


def generate_ts_indicators(data, warmup_years=3, years_offset=3):
    """
    Generate the normalized indicators for a time series.

    :param data: pd.DataFrame, dataframe to extract data from
    :param warmup_years: int, number of years that represent warmup period
    :param years_offset: int, number of years to offset by

    :returns: tuple[List], time index and values
    """
    index = []
    values = []

    offset_dt = pd.DateOffset(years=years_offset - 1)

    # Initial datetimes
    past_date = data.index[0] + pd.DateOffset(years=warmup_years)
    curr_date = past_date + offset_dt
    start_indicat = data[past_date:curr_date].sum()
    while(curr_date < data.index[-1]):

        index.append(curr_date)
        indicat = data[past_date:curr_date].sum() / start_indicat if start_indicat > 0.0 else 0.0
        values.append(indicat)

        curr_date = curr_date + pd.DateOffset(years=1)
        past_date = curr_date - offset_dt
    # End while

    return index, values
# End generate_ts_indicators()


def calc_perc_change(series):
    """Calculate percent change from first value in a series

    :param series: pd.Series, array of values

    :returns: pd.Series, of percentage change from first value in series.
    """
    import warnings
    msg = """
    This currently calculates relative change from first series.
    We'd want this to display relative change from historic average!
    """
    warnings.warn(msg, FutureWarning)
    first_val = series[0]
    denom = (first_val * (first_val / abs(first_val)))
    return series.apply(lambda x: ((x - first_val) / denom) * 100.0)
# End calc_perc_change()


def convert_indicators_to_pd(x):
    """Convert CIM result dict to a DataFrame.
    
    Parameters
    ==========
    * x : dict
    """
    master = pd.DataFrame()
    for k, v in x.items():
        try:
            tmp = pd.DataFrame.from_dict(v).T
        except ValueError:
            tmp = pd.DataFrame(v, index=[k]).T
        # End try

        master = pd.concat((master, tmp), axis=1)
    # End for

    return master


def combine_result_dfs(x, y):
    """Combine two CIM result DFs, adding rows together."""
    return pd.concat((x, y), ignore_index=True)


def rename_target_scenarios(df):
    """Modify climate scenario names.
    
    Indicates which scenarios were of the most interest for the study.
    """
    tgt_col = 'names'
    if 'climate_scenario' in df.columns:
        tgt_col = 'climate_scenario'
    df.loc[df[tgt_col] == 'worst_case_rcp45_2016-2045',
           tgt_col] = 'worst_case_rcp45_2016-2045 (dry)'
    df.loc[df[tgt_col] == 'best_case_rcp45_2016-2045',
           tgt_col] = 'best_case_rcp45_2016-2045 (usual)'
    df.loc[df[tgt_col] == 'best_case_rcp45_2036-2065',
           tgt_col] = 'best_case_rcp45_2036-2065 (wet)'
    return df


def remove_matches(df, tgt_cols):
    """Remove matching columns by partial match.
    
    Parameters
    ==========
    * df : Pandas DataFrame
    * tgt_cols: list[str], of partial column names to match on
    """
    for tc in tgt_cols:
        df = df[[c for c in df.columns if tc not in c]]

    return df
# End remove_matches()


def to_category(df, colname, vals):
    """Convert column to categoricals with specified values
    
    Parameters
    ----------
    * df : dataframe
    * colname : str, name of column
    * vals : CategoricalDtype
    
    Returns
    ----------
    * dataframe
    """
    df[colname] = df[colname].astype(vals).astype('category')
    return df
