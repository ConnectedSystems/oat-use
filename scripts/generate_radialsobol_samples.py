import pandas as pd
from SALib.sample.radial.radial_sobol import sample

from .settings import *  # import project-specific settings


# read in previous sample set for a single climate scenario
# we use this as a template
df = pd.read_csv(indir+'example_sample.csv', index_col=0)

is_perturbed = (df != df.iloc[0]).any()
perturbed_cols = df.loc[:, is_perturbed].columns
constant_cols = df.loc[:, ~is_perturbed].columns

cat_cols = df.select_dtypes('object').columns
p_df = df.copy()

for col in cat_cols:
    p_df[col] = p_df[col].astype('category')


perturbed = p_df.loc[:, is_perturbed]
for col in perturbed_cols:
    if col in cat_cols:
        perturbed[col] = perturbed[col].cat.codes


bounds = list(zip(perturbed.min().tolist(), perturbed.max().tolist()))
problem = {
    'num_vars': len(perturbed_cols),
    'names': perturbed_cols,
    'bounds': bounds
}


# Create Saltelli samples
# SALib expects purely numeric values so categoricals are transformed as such
samples = sample(problem, 10, seed=101)

# Create template
rsobol_df = df.iloc[0][constant_cols].to_frame().T
rsobol_df = pd.concat([rsobol_df]*len(samples), ignore_index=True)

df_samples = pd.DataFrame(data=samples, columns=perturbed_cols)

# Export numeric sample values
rsobol_df[df_samples.columns] = df_samples

numeric_df = rsobol_df.copy()
for col in numeric_df:
    if col in cat_cols:
        numeric_df[col] = numeric_df[col].astype('category')
        numeric_df[col] = numeric_df[col].cat.codes

numeric_df.to_csv(indir+'radial_10_numeric_samples.csv')


# Create dataframe of perturbed values
# Numeric values are mapped to their categorical representations
# where appropriate. This may affect the sampling.
for col in df_samples:
    if col in cat_cols:
        cats = p_df[col].cat.categories
        df_samples[col] = pd.cut(df_samples[col], len(cats), labels=cats.tolist())


# Replace template values with perturbed values and export to csv
rsobol_df[df_samples.columns] = df_samples
rsobol_df = rsobol_df.rename('{}_radial_10_sample'.format)
rsobol_df.to_csv(indir+'radial_10_samples.csv')
