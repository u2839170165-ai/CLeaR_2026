import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from time import sleep

from benchmark_CD import causallearn_benchmark_
from utils import save_pickle, heterogeneous_knn_imputation
import argparse


if __name__ == '__main__':
    # parse CLI
    parser = argparse.ArgumentParser(description='Run causallearn benchmarks on selected dataset')
    parser.add_argument('--dataset', choices=['tumor', 'full', 'full_wo_tmb'], default='tumor',
                        help="Choose 'tumor' to run benchmarks on tumor datasets or 'full' for the full dataset collection")
    args = parser.parse_args()

    print(f"Running benchmarks for dataset: {args.dataset}")

    # import preprocessed data
    data_dir = Path(__file__).resolve().parent/'datasets'

    # Select lists depending on requested dataset
    if args.dataset == 'tumor':
        Tumor_data_mixed = pd.read_csv(data_dir/'Tumor_data_mixed.csv', sep='\t').drop(['PatientID'], axis=1, errors='ignore')
        Tumor_data_discrete = pd.read_csv(data_dir/'Tumor_data_discrete.csv', sep='\t').drop(['PatientID'], axis=1, errors='ignore')
        key_list = ['Tumor_data_mixed', 'Tumor_data_mixed_no_nan', 'Tumor_data_mixed_imputed', 'Tumor_data_discrete_imputed']
        out_dir = Path(__file__).resolve().parent/'results/tumor_data'
        data_list = [Tumor_data_mixed, Tumor_data_mixed, Tumor_data_mixed, Tumor_data_discrete]
        is_discrete_list = [False, False, False, True]
        is_complete_list = [False, True, True, True]
        is_imputed_list = [False, False, True, True]
    elif args.dataset == 'full':  # full
        Full_continuous = pd.read_csv(data_dir/'Full_continuous.csv', sep='\t').drop(['PatientID'], axis=1, errors='ignore')
        Full_discrete = pd.read_csv(data_dir/'Full_discrete.csv', sep='\t').drop(['PatientID'], axis=1, errors='ignore')
        key_list = ['Full_continuous', 'Full_continuous_no_nan', 'Full_continuous_imputed']
        out_dir = Path(__file__).resolve().parent/'results/full_data'
        data_list = [Full_continuous, Full_continuous, Full_continuous]
        is_discrete_list = [False, False, False]
        is_complete_list = [False, True, True]
        is_imputed_list = [False, False, True]
    elif args.dataset == 'full_wo_tmb':  # full without TMB
        Full_continuous_wo_tmb = pd.read_csv(data_dir/'Full_continuous_wo_tmb.csv', sep='\t').drop(['PatientID'], axis=1, errors='ignore')
        key_list = ['Full_continuous_wo_tmb', 'Full_continuous_wo_tmb_no_nan', 'Full_continuous_wo_tmb_imputed']
        out_dir = Path(__file__).resolve().parent/'results/full_data_wo_tmb'
        data_list = [Full_continuous_wo_tmb, Full_continuous_wo_tmb, Full_continuous_wo_tmb]
        is_discrete_list = [False, False, False]
        is_complete_list = [False, True, True]
        is_imputed_list = [False, False, True]

    # Ensure out_dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if output files already exist
    for key in key_list:
        file_path = Path(out_dir/f"Outputs_{key}.pkl")
        if file_path.is_file():
            print(f"{file_path.name} already exists. Loading existing results.")
            exit(1)
    
    def benchmark(index):  
        dataset = data_list[index]
        if index == 1 : dataset = dataset.dropna()
        elif index == 2 : dataset = heterogeneous_knn_imputation(dataset, continuous_cols=['Age', 'SPY'], out_dir=out_dir, file_name='distribution_mixed')
        elif index == 3 : dataset = heterogeneous_knn_imputation(dataset, continuous_cols = [], discrete_cols = dataset.columns.tolist(), out_dir=out_dir, file_name='distribution_discrete')
        is_discrete = is_discrete_list[index]
        is_complete = is_complete_list[index]
        is_imputed = is_imputed_list[index]
        Outputs = causallearn_benchmark_(data=dataset, is_discrete=is_discrete, is_complete=is_complete, is_imputed=is_imputed)
        sleep(0.1)  # Ensure different timestamps if needed
        save_pickle(Outputs, out_dir/f"Outputs_{key_list[index]}.pkl")
        print(Outputs)

    results = Parallel(n_jobs=len(data_list))(delayed(benchmark)(index) for index in range(len(data_list)))


        
