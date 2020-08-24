import pandas as pd


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
# End convert_indicators_to_pd
