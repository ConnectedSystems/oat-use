import pandas as pd

from SALib.analyze.radial_ee import analyze as ee_analyze
from SALib.analyze.sobol_jansen import analyze as jansen_analyze
from SALib.plotting.bar import plot as barplot

# results produced with
# python launch.py --specific_inputs oat_mc_10_samples.csv --num_cores 48
# python launch.py --specific_inputs oat_cim_extremes.csv --num_cores 2

# python launch.py --specific_inputs moat_10_samples.csv --num_cores 46

from .settings import *


data_dir = indir

problem = {
    'num_vars': 53,
    'names': ['Farm___Crops___variables___Dryland_Winter_Barley___root_depth_m',
        'Farm___Crops___variables___Dryland_Winter_Barley___water_use_ML_per_Ha',
        'Farm___Crops___variables___Dryland_Winter_Barley___yield_per_Ha',
        'Farm___Crops___variables___Dryland_Winter_Canola___root_depth_m',
        'Farm___Crops___variables___Dryland_Winter_Canola___water_use_ML_per_Ha',
        'Farm___Crops___variables___Dryland_Winter_Canola___yield_per_Ha',
        'Farm___Crops___variables___Dryland_Winter_Wheat___root_depth_m',
        'Farm___Crops___variables___Dryland_Winter_Wheat___water_use_ML_per_Ha',
        'Farm___Crops___variables___Dryland_Winter_Wheat___yield_per_Ha',
        'Farm___Crops___variables___Irrigated_Winter_Barley___root_depth_m',
        'Farm___Crops___variables___Irrigated_Winter_Barley___water_use_ML_per_Ha',
        'Farm___Crops___variables___Irrigated_Winter_Barley___yield_per_Ha',
        'Farm___Crops___variables___Irrigated_Winter_Canola___root_depth_m',
        'Farm___Crops___variables___Irrigated_Winter_Canola___water_use_ML_per_Ha',
        'Farm___Crops___variables___Irrigated_Winter_Canola___yield_per_Ha',
        'Farm___Crops___variables___Irrigated_Winter_Wheat___root_depth_m',
        'Farm___Crops___variables___Irrigated_Winter_Wheat___water_use_ML_per_Ha',
        'Farm___Crops___variables___Irrigated_Winter_Wheat___yield_per_Ha',
        'Farm___Fields___soil___zone_10___TAW_mm',
        'Farm___Fields___soil___zone_11___TAW_mm',
        'Farm___Fields___soil___zone_12___TAW_mm',
        'Farm___Fields___soil___zone_1___TAW_mm',
        'Farm___Fields___soil___zone_2___TAW_mm',
        'Farm___Fields___soil___zone_3___TAW_mm',
        'Farm___Fields___soil___zone_4___TAW_mm',
        'Farm___Fields___soil___zone_5___TAW_mm',
        'Farm___Fields___soil___zone_6___TAW_mm',
        'Farm___Fields___soil___zone_7___TAW_mm',
        'Farm___Fields___soil___zone_8___TAW_mm',
        'Farm___Fields___soil___zone_9___TAW_mm',
        'Farm___Irrigations___Gravity___cost_per_Ha',
        'Farm___Irrigations___Gravity___head_pressure',
        'Farm___Irrigations___Gravity___irrigation_efficiency',
        'Farm___Irrigations___Gravity___pumping_cost_per_ML',
        'Farm___Irrigations___PipeAndRiser___cost_per_Ha',
        'Farm___Irrigations___PipeAndRiser___head_pressure',
        'Farm___Irrigations___PipeAndRiser___irrigation_efficiency',
        'Farm___Irrigations___PipeAndRiser___pumping_cost_per_ML',
        'Farm___Irrigations___Spray___cost_per_Ha',
        'Farm___Irrigations___Spray___head_pressure',
        'Farm___Irrigations___Spray___irrigation_efficiency',
        'Farm___Irrigations___Spray___pumping_cost_per_ML',
        'Farm___zone_10___Irrigation', 'Farm___zone_11___Irrigation',
        'Farm___zone_2___Irrigation', 'Farm___zone_4___Irrigation',
        'Farm___zone_6___Irrigation', 'Farm___zone_7___Irrigation',
        'Farm___zone_8___Irrigation', 'Farm___zone_9___Irrigation',
        'policy___goulburn_allocation_scenario', 'policy___gw_cap',
        'policy___gw_restriction'],
    'bounds': [(0.80008164104, 1.49988829764),
    (1.50055050742, 2.99888102069),
    (1.5019032420200003, 3.4997506932099998),
    (0.800586478968, 1.4996985073),
    (2.50048002895, 5.9984797603299995),
    (0.801052350325, 2.59824297051),
    (0.800504246618, 1.49975544648),
    (2.5014981435299997, 5.9979681912),
    (1.5004709810799999, 5.99716646463),
    (0.800280272497, 1.49937425734),
    (1.5009590614, 2.9992559947000004),
    (2.50329796931, 6.996816011819999),
    (0.800211596215, 1.49974890273),
    (2.0025975557, 5.99742468979),
    (1.3008100600299999, 4.99958661017),
    (0.8000586077680001, 1.7993585851400002),
    (2.50005748529, 5.99920182664),
    (1.5021921746899998, 7.99719295089),
    (150.013080285, 199.99630294),
    (145.01266211, 184.97447762599998),
    (145.036691741, 184.96132256099997),
    (145.017973816, 184.964659778),
    (145.009985077, 184.987775366),
    (100.017759932, 159.950281059),
    (100.00893349, 159.939807798),
    (150.002663759, 199.995911171),
    (150.049539279, 199.966206716),
    (75.011883698, 109.982509833),
    (100.007801344, 159.986958043),
    (145.015806747, 184.983072651),
    (2000.04766978, 2499.9660698000002),
    (8.00489093285, 14.999582054100001),
    (0.500092622216, 0.8998440697460001),
    (8.0072724319, 14.9995752798),
    (2000.65212205, 3299.41488388),
    (8.00365090987, 14.9983740134),
    (0.600018657025, 0.899703908987),
    (8.005434387660001, 14.9933485659),
    (2500.62094903, 3499.76177012),
    (25.0039236705, 34.9957834096),
    (0.7001056060199999, 0.8998137827079999),
    (30.000316497100002, 59.9914045149),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 2.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 2.0),
    (0.0, 2.0),
    (0.600156362739, 0.999676343195),
    (0.0, 1.0)]
}


def collect_results(problem, oat_length, reps, np_res, numeric_vals):
    jansen_results_df = pd.DataFrame()
    ee_results_df = pd.DataFrame()

    rep_length = oat_length * reps

    _, cols = np_res.shape

    for col in range(cols):
        cn = col_names[col]
        res = np_res[:rep_length, col]

        si = jansen_analyze(problem, res, reps, seed=101)

        js_df = si.to_df()
        js_df.columns = ['{}_{}'.format(cn, suf) for suf in js_df.columns]
        jansen_results_df = pd.concat([jansen_results_df, js_df], axis=1)

        si = ee_analyze(problem, numeric_vals[:rep_length],
                        res, reps, seed=101)
        ee_df = si.to_df()
        ee_df.columns = ['{}_{}'.format(cn, suf) for suf in ee_df.columns]
        ee_results_df = pd.concat([ee_results_df, ee_df], axis=1)
    
    return jansen_results_df, ee_results_df
# End collect_results()


def plot_results(jansen_results_df, ee_results_df, target_metric):

    # STs = [c for c in jansen_results_df.columns if '_conf' not in c and target_metric in c]
    idx = [True if 'irrigation' in r.lower() else False for r in jansen_results_df.index]

    # ax = jansen_results_df.loc[idx, STs].plot(kind='bar', figsize=(10,6))
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    tgt_idx = [c for c in ee_results_df.columns if target_metric.lower() in c.lower()]
    ax = ee_results_df.loc[idx, tgt_idx].plot(kind='bar', figsize=(10,6))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# End plot_results()

template_df = pd.read_csv(f'{data_dir}example_sample.csv', index_col=0)
is_perturbed = (template_df != template_df.iloc[0]).any()
perturbed_cols = template_df.loc[:, is_perturbed].columns

target_num_vars = problem['num_vars']
oat_length = target_num_vars + 1
target_metric = "SW Allocation Index"


### Extreme values without interactions ###

numeric_samples = pd.read_csv(f'{data_dir}extreme_numeric_samples.csv', index_col=0)
numeric_samples = numeric_samples[perturbed_cols]
numeric_vals = numeric_samples.values

extreme_results = pd.read_csv(f'{data_dir}no_irrigation_extreme_results.csv', index_col=0)
np_res = extreme_results.values
col_names = extreme_results.columns

extreme_results = {}
for i in range(len(col_names)):
    x_diff = (numeric_vals[0, :] - numeric_vals[1, :])
    y_diff = (np_res[0, i] - np_res[1, i])

    extreme_results[col_names[i]] = y_diff / x_diff
# End for

no_ext_results = pd.DataFrame(extreme_results, index=perturbed_cols).T
no_ext_results.columns = [c.replace('Farm___Irrigations___', '') for c in no_ext_results.columns]
tgt_cols = [c for c in no_ext_results.columns if 'gravity___irrigation_efficiency' in c.lower()]

# no_ext_results.loc[tgt_idx, tgt_cols].plot(kind='bar', legend=None)

### Extremes with interactions ###

extreme_results = pd.read_csv(f'{data_dir}with_irrigation_extreme_results.csv', index_col=0)
np_res = extreme_results.values
col_names = extreme_results.columns

extreme_results = {}
for i in range(len(col_names)):
    x_diff = (numeric_vals[0, :] - numeric_vals[1, :])
    y_diff = (np_res[0, i] - np_res[1, i])

    extreme_results[col_names[i]] = abs(y_diff / x_diff)
# End for

with_ext_results = pd.DataFrame(extreme_results, index=perturbed_cols).T
tgt_idx = [c for c in with_ext_results.index if target_metric.lower() in c.lower()]

with_ext_results.columns = [c.replace('Farm___Irrigations___', '') for c in with_ext_results.columns]
tgt_cols = [c for c in with_ext_results.columns if 'gravity___irrigation_efficiency' in c.lower()]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)

no_ext_results.loc[tgt_idx, tgt_cols].plot(kind='bar', legend=None, 
                                           title='Disabled interactions',
                                           ax=axes[0])
axes[0].set_ylabel('Absolute Change')

with_ext_results.loc[tgt_idx, tgt_cols].plot(kind='bar', 
                                             legend=None,
                                             title='Enabled interactions',
                                             ax=axes[1]).legend(
                                                bbox_to_anchor=(1.75, 0.65)
                                             )
fig.suptitle("Extremity Testing", x=0.4, y=1.05, fontsize=14)
fig.tight_layout()
fig.savefig(f'{fig_dir}extremity_testing_results.png', dpi=300, bbox_inches='tight')


### Larger samples

# Without irrigation interaction with SW model

numeric_samples = pd.read_csv(f'{data_dir}oat_mc_10_numeric_samples.csv', index_col=0)
numeric_samples = numeric_samples[perturbed_cols]
numeric_vals = numeric_samples.values

oat_10_no_irrigation_results = pd.read_csv(f'{data_dir}oat_no_irrigation_10_results.csv', index_col=0)
np_res = oat_10_no_irrigation_results.values

mu_star_col = target_metric + '_mu_star'
sigma_col = target_metric + '_sigma'

fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)

res = {'mu_star': {}, 'sigma': {}}
tgt_param = 'Farm___Irrigations___Gravity___irrigation_efficiency'
for reps in range(1, 11):
    jansen_results_df, ee_results_df = collect_results(problem, oat_length, reps, np_res, numeric_vals)

    runs = reps * oat_length

    res['mu_star'][runs] = ee_results_df.loc[tgt_param, mu_star_col]
    res['sigma'][runs] = ee_results_df.loc[tgt_param, sigma_col]

oat_no_interaction = pd.DataFrame(data=res)
oat_no_interaction.plot(kind='bar', ax=axes[0], title='Disabled Interactions')


# With irrigation interaction with SW model

oat_10_with_irrigation_results = pd.read_csv(f'{data_dir}oat_with_irrigation_10_results.csv', index_col=0)
np_res = oat_10_with_irrigation_results.values

res = {'mu_star': {}, 'sigma': {}}
for reps in range(1, 11):
    jansen_results_df, ee_results_df = collect_results(problem, oat_length, reps, np_res, numeric_vals)

    runs = reps * oat_length

    res['mu_star'][runs] = ee_results_df.loc[tgt_param, mu_star_col]
    res['sigma'][runs] = ee_results_df.loc[tgt_param, sigma_col]

oat_with_interaction = pd.DataFrame(data=res)
oat_with_interaction.plot(kind='bar', ax=axes[1], title='Enabled Interactions')
fig.suptitle('Sensitivity of SW Allocation to\nGravity Irrigation Efficiency',
             y=1.05, fontsize=14)
fig.tight_layout()

fig.savefig(f'{fig_dir}radial_oat_results.png', dpi=300, bbox_inches='tight')