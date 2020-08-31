import numpy as np
import pandas as pd
from SALib.sample.radial.radial_mc import sample as mc_sampler
from .settings import *  # import project-specific settings


inbounds = pd.read_csv(indir+'2019-11-06_012547_272b10e9-6383-5514-89bb-6403d8f8c0a7_param_bounds.dat', index_col=0)

# read in previous sample set for a single climate scenario
# we use this as a template
df = pd.read_csv(indir+'example_sample.csv', index_col=0)

is_perturbed = (df != df.iloc[0]).any()
perturbed_cols = df.loc[:, is_perturbed].columns
constant_cols = df.loc[:, ~is_perturbed].columns

cat_cols = df.select_dtypes('object').columns
p_df = df.copy()

extremes = p_df.iloc[0:2].copy()

col_idx = np.where(is_perturbed)[0]
extremes.iloc[0, col_idx] = p_df.loc[:, is_perturbed].min()
extremes.iloc[1, col_idx] = p_df.loc[:, is_perturbed].max()
extremes.to_csv(indir+"extremes_samples.csv")

for col in cat_cols:
    p_df[col] = p_df[col].astype('category')
    extremes[col] = extremes[col].astype('category')


perturbed = p_df.loc[:, is_perturbed]

for col in perturbed_cols:
    if col in cat_cols:
        perturbed[col] = perturbed[col].cat.codes
        

for col in extremes.columns:
    if col in cat_cols:
        extremes[col] = extremes[col].cat.codes

# export numeric values
extremes.to_csv(indir+"extreme_numeric_samples.csv")

bounds = list(zip(perturbed.min().tolist(), perturbed.max().tolist()))

problem = {
    'num_vars': len(perturbed_cols),
    'names': perturbed_cols,
    'bounds': bounds
}

# Create MC samples (p+1)*n
# SALib expects purely numeric values so categoricals are transformed as such
oat_samples = mc_sampler(problem, 10, seed=101)

# Create template
oat_mc_df = df.iloc[0][constant_cols].to_frame().T
oat_mc_df = pd.concat([oat_mc_df]*len(oat_samples), ignore_index=True)

df_samples = pd.DataFrame(data=oat_samples, columns=perturbed_cols)

# Export numeric sample values
oat_mc_df[df_samples.columns] = df_samples

numeric_df = oat_mc_df.copy()
for col in numeric_df:
    if col in cat_cols:
        numeric_df[col] = numeric_df[col].astype('category')
        numeric_df[col] = numeric_df[col].cat.codes

numeric_df.to_csv(indir+'oat_mc_10_numeric_samples.csv')

# Create dataframe of perturbed values
# Numeric values are mapped to their categorical representations
# where appropriate. This may affect the sampling.
for col in df_samples:
    if col in cat_cols:
        cats = p_df[col].cat.categories
        df_samples[col] = pd.cut(df_samples[col], len(cats), labels=cats.tolist())


# Replace template values with perturbed values and export to csv
oat_mc_df[df_samples.columns] = df_samples

oat_mc_df = oat_mc_df.rename('{}_mc_sample'.format)

oat_mc_df.to_csv(indir+'oat_mc_10_samples.csv')
