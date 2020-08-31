import numpy as np
import pandas as pd

from SALib.analyze.rbd_fast import analyze as rbd_analyze

from .settings import *  # import project-specific settings


data_dir = indir


### Define problem spec

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

target_param = "Farm___Irrigations___Gravity___irrigation_efficiency"
target_metric = "SW Allocation Index"


template_df = pd.read_csv(f'{data_dir}example_sample.csv', index_col=0)
is_perturbed = (template_df != template_df.iloc[0]).any()
perturbed_cols = template_df.loc[:, is_perturbed].columns

target_num_vars = problem['num_vars']
oat_length = target_num_vars + 1

### Analyze extreme results

extreme_numeric_samples = pd.read_csv(f'{data_dir}extreme_numeric_samples.csv', index_col=0)
extreme_numeric_samples = extreme_numeric_samples[perturbed_cols]
extreme_numeric_vals = extreme_numeric_samples.values

# Extreme values without interactions
extreme_results = pd.read_csv(f'{data_dir}no_irrigation_extreme_results.csv', index_col=0)
extreme_disabled_res = extreme_results.values
col_names = extreme_results.columns

target_result_idx = extreme_results.columns.tolist().index(target_metric)

rbd_fast_results = rbd_analyze(problem, extreme_numeric_vals, 
                               extreme_disabled_res[:, target_result_idx], 
                               M=4, seed=101)
rbd_fast_results.plot()

# Extreme values with interactions
extreme_results = pd.read_csv(f'{data_dir}with_irrigation_extreme_results.csv', index_col=0)
extreme_enabled_res = extreme_results.values

rbd_fast_results = rbd_analyze(problem, extreme_numeric_vals, 
                               extreme_enabled_res[:, target_result_idx], 
                               M=4, seed=101)
rbd_fast_results.plot()


### With more samples...

numeric_samples = pd.read_csv(f'{data_dir}oat_mc_10_numeric_samples.csv', index_col=0)
numeric_samples = numeric_samples[perturbed_cols]
numeric_vals = numeric_samples.values

# Add extreme sample points
# numeric_vals = np.insert(numeric_vals, 0, extreme_numeric_vals, axis=0)


# Coupling disabled
oat_10_no_irrigation_results = pd.read_csv(f'{data_dir}oat_no_irrigation_10_results.csv', index_col=0)
np_res = oat_10_no_irrigation_results.values

# Add extremes
# np_res = np.insert(np_res, 0, extreme_disabled_res, axis=0)

target_result_idx = oat_10_no_irrigation_results.columns.tolist().index(target_metric)

res = []
idx = []
for reps in range(2, 150):
    try:
        rbd_fast_results = rbd_analyze(problem, numeric_vals[:reps, ], 
                                       np_res[:reps, target_result_idx], 
                                       M=4, seed=101)
    except ZeroDivisionError:
        res.append(np.nan)
        idx.append(reps)
        continue

    disabled = rbd_fast_results.to_df()
    tmp = disabled.loc[target_param, 'S1']  # .plot(kind='bar')
    res.append(tmp)
    idx.append(reps)

pd.DataFrame({'S1':res}, index=idx).plot(kind='bar')


# Coupling enabled

oat_10_with_irrigation_results = pd.read_csv(f'{data_dir}oat_with_irrigation_10_results.csv', index_col=0)
np_res = oat_10_with_irrigation_results.values

# Add extremes
# np_res = np.insert(np_res, 0, extreme_enabled_res, axis=0)

res = []
idx = []
for reps in range(2, 150):
    try:
        rbd_fast_results = rbd_analyze(problem, numeric_vals[:reps, ], 
                                       np_res[:reps, target_result_idx], 
                                       M=4, seed=101)
    except ZeroDivisionError:
        res.append(np.nan)
        idx.append(reps)
        continue

    enabled = rbd_fast_results.to_df()
    tmp = enabled.loc[target_param, 'S1']
    res.append(tmp)
    idx.append(reps)

pd.DataFrame({'S1':res}, index=idx).plot(kind='bar')



### Targeted analysis

numeric_samples = pd.read_csv(f'{data_dir}oat_mc_10_numeric_samples.csv', index_col=0)
numeric_samples = numeric_samples[perturbed_cols]
numeric_vals = numeric_samples.values

unique_vals, unique_idx = np.unique(numeric_samples[target_param], return_index=True)

# Coupling disabled

oat_10_no_irrigation_results = pd.read_csv(f'{data_dir}oat_no_irrigation_10_results.csv', index_col=0)
np_res = oat_10_no_irrigation_results.values

target_result_idx = oat_10_no_irrigation_results.columns.tolist().index(target_metric)

numeric_vals = numeric_vals[unique_idx]
np_res = np_res[unique_idx]

# Add extreme sample points
# numeric_vals = np.insert(numeric_vals, 0, extreme_numeric_vals, axis=0)

# Add extreme results
# np_res = np.insert(np_res, 0, extreme_disabled_res, axis=0)

res = []
idx = []
for reps in range(2, numeric_vals.shape[0]):

    try:
        rbd_fast_results = rbd_analyze(problem, numeric_vals[:reps, ], 
                                       np_res[:reps, target_result_idx], 
                                       M=4, seed=101)
    except ZeroDivisionError:
        res.append(np.nan)
        idx.append(reps)
        continue

    disabled = rbd_fast_results.to_df()
    tmp = disabled.loc[target_param, 'S1']
    res.append(tmp)
    idx.append(reps)

pd.DataFrame({'S1':res}, index=idx).plot(kind='bar')


# Coupling enabled

oat_10_with_irrigation_results = pd.read_csv(f'{data_dir}oat_with_irrigation_10_results.csv', index_col=0)
np_res = oat_10_with_irrigation_results.values

np_res = np_res[unique_idx]

# Add extreme results
# np_res = np.insert(np_res, 0, extreme_enabled_res, axis=0)

res = []
idx = []
for reps in range(2, len(unique_vals)):
    try:
        rbd_fast_results = rbd_analyze(problem, numeric_vals[:reps, ], 
                                       np_res[:reps, target_result_idx], 
                                       M=4, seed=101)
    except ZeroDivisionError:
        res.append(np.nan)
        idx.append(reps)
        continue

    enabled = rbd_fast_results.to_df()
    tmp = enabled.loc[target_param, 'S1']
    res.append(tmp)
    idx.append(reps)

pd.DataFrame({'S1':res}, index=idx).plot(kind='bar')