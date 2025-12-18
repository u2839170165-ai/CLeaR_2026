import pandas as pd
from pathlib import Path
from utils import discretize_dataframe, merge_dataframes, selective_fillna, str_to_class
# Load datasets

# Read all CSV files in Datasets except CombinedDF.csv into a list and a dict
data_dir = Path(__file__).resolve().parent/'datasets'
csv_files = sorted([p for p in data_dir.glob('*.csv') if p.name != 'CombinedDF.csv'])

dataframes = []           # list of DataFrame objects
dataframe_names = []      # corresponding file base names
dataframes_by_name = {}   # optional dict name -> DataFrame

for p in csv_files:
    # read each CSV with tab separator
    df = pd.read_csv(p, sep='\t')
    dataframes.append(df)
    name = p.stem
    dataframe_names.append(name)
    dataframes_by_name[name] = df
    # print(f'Loaded {name} with shape {df.shape}')
    # print(df.columns)

# Load CombinedDF separately if present
combined_path = data_dir/'CombinedDF.csv'
CombinedDF = pd.read_csv(combined_path, sep='\t') if combined_path.exists() else None

Common_cols=dataframes[0].columns.intersection(dataframes[1].columns).tolist()
# print(f'Common columns between first two dataframes: {Common_cols}')

Combined = merge_dataframes(dataframes,Common_cols)
# Remove duplicate rows if any
Combined = Combined.drop_duplicates()
# Check for discrepancies between CombinedDF and Combined
IDs1, IDs2 = set(CombinedDF["Unnamed: 0"].unique()), set(Combined.PatientID.unique())
# print('IDs in CombinedDF not in Combined:', IDs1 - IDs2)
# print('IDs in Combined not in CombinedDF:', IDs2 - IDs1)

Tumor_vars = ['Age', 'Sex', 'Smoking', 'SPY', 'Stage', 'Status', 'TMB']
Discretized_vars = ['Age_Discrete', 'SPY_Discrete', 'TMB_Discrete', 'Tstage']  

# Apply the conversions to Combined (use PatientID as index if present)
CombinedDF.index.name == 'Unnamed: 0'
Combined.index.name == 'PatientID'

Combined = selective_fillna(Combined, Discretized_vars+Tumor_vars)

Discrete_petient_data = Combined[Discretized_vars]
Full_continuous = Combined.drop(columns = Discretized_vars)
Tumor_data = Full_continuous[Tumor_vars]

# --- added: mappings and helper to convert strings to classes / categorical and set index ---
sex_map = {'Male': 0, 'Female': 1}
smoking_map = {
    'Current Smoker': 4,
    'Reformed >15': 2,
    'Reformed <=15': 3,
    'Reformed': 1,
    'Never Smoke': 0
}
status_map = {'0:LIVING': 0, '1:DECEASED': 1}
stage_map = {'IA': 0, 'IB': 1, 'IIA': 2, 'IIB': 3, 'IIIA': 4, 'IIIB': 5, 'IV': 6}

age_map = {
    '38-43': 0,
    '43-48': 1,
    '48-54': 2,
    '54-59': 3,
    '59-64': 4,
    '64-69': 5,
    '69-74': 6,
    '74-80': 7,
    '80-85': 8,
    '85-90': 9
}

spy_map = {
    '0-15': 0,
    '15-29': 1,
    '29-44': 2,
    '44-59': 3,
    '59-74': 4,
    '74-88': 5,
    '88-103': 6,
    '103-118': 7,
    '118-132': 8,
    '132-147': 9
}

tmb_map = {
    '0-4': 0,
    '4-8': 1,
    '8-13': 2,
    '13-17': 3,
    '17-21': 4,
    '21-25': 5,
    '25-29': 6,
    '29-34': 7,
    '34-38': 8,
    '38-42': 9
}

Tumor_data_1 = str_to_class(
    Tumor_data,
    mappings={
        'Sex': sex_map,
        'Smoking': smoking_map,
        'Status': status_map,
        'Stage': stage_map
    },
    categorical_cols=None,
    inplace=False
)

Discrete_petient_data_1 = str_to_class(
    Discrete_petient_data,
    mappings={
        'Age_Discrete': age_map,
        'SPY_Discrete': spy_map,
        'TMB_Discrete': tmb_map
    },
    categorical_cols=None,
    inplace=False
)

Tumor_data_2 = Tumor_data_1.copy()
Tumor_data_2[['Age', 'SPY', 'TMB']] = Discrete_petient_data_1[['Age_Discrete', 'SPY_Discrete', 'TMB_Discrete']]
Full_continuous[Tumor_vars] = Tumor_data_1
Full_discrete = Full_continuous.copy()
Full_discrete[['Age', 'SPY', 'TMB']] = Discrete_petient_data_1[['Age_Discrete', 'SPY_Discrete', 'TMB_Discrete']]
cols_to_discretize = set(Full_discrete.columns).difference(set(Tumor_vars+['PatientID']))
Full_discrete = discretize_dataframe(Full_discrete, cols_to_discretize)
Full_continuous_wo_tmb = Full_continuous.copy().drop(columns='TMB')

# Save processed DataFrames to CSV files
Full_continuous.to_csv(data_dir/'Full_continuous.csv', sep = '\t', index = False)
Full_discrete.to_csv(data_dir/'Full_discrete.csv', sep = '\t', index = False)
Tumor_data_1.to_csv(data_dir/'Tumor_data_mixed.csv', sep = '\t', index = False)
Tumor_data_2.to_csv(data_dir/'Tumor_data_discrete.csv', sep = '\t', index = False)
Full_continuous_wo_tmb.to_csv(data_dir/'Full_continuous_wo_tmb.csv', sep = '\t', index = False)

