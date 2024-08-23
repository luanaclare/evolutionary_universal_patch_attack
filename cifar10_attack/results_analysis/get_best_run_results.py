import os
import pandas as pd


scenarios = ['f1', 'f2', 'f3']
modelNames = {
    'VGG16': ['VGG16__run__2024_06_26__06_36_09_480__112681534716641280', 'VGG16__run__2024_07_09__10_05_07_686__112755966455709696', 'VGG16__run__2024_07_01__20_38_08_586__112713157088772096'],
    'ResNet50': ['ResNet50__run__2024_06_21__22_37_27_519__112657003152605184', 'ResNet50__run__2024_07_28__10_27_02_652__112863636530921472', 'ResNet50__run__2024_07_25__12_01_58_461__112847022880260096']
}

# Define the columns for the results
columns = [
    'Scenario', 'Model', 'Best Fitness Final', 'Best ASR Evolution', 'Best ASR Test',
    'Best Linf Evolution', 'Best Linf Test', 'Best L2 Evolution', 'Best L2 Test', 'Best ACC Old', 'Best ACC New',
    'Best Time (min)', 'Best Folder', 'Best Seed'
]

results_data = []

for model_name, best_runs in modelNames.items():
    results_data = []
    
    for scenario, bestrun in zip(scenarios, best_runs):
        best_folder_path = f'../runs_final/{scenario}/{bestrun}'  # Path for the best run

        if os.path.isdir(best_folder_path):
            logs_folder_path = os.path.join(best_folder_path, 'logs')
            if os.path.isdir(logs_folder_path):
                for file_name in os.listdir(logs_folder_path):
                    if file_name.startswith('timings') and file_name.endswith('.csv'):
                        file_path = os.path.join(logs_folder_path, file_name)
                        df = pd.read_csv(file_path)
                        time_value = df[' total'].values
                        time_min = time_value[0] / 60  # Convert to minutes

            best_info_file = os.path.join(best_folder_path, 'best_info.csv')
            df = pd.read_csv(best_info_file)
            best_fitness = df['fitness'].values[0]
            asr_evolution = df['adv_quantity'].values[0]

            test_info_file = os.path.join(best_folder_path, 'test_info.csv')
            df = pd.read_csv(test_info_file)
            asr_test = df['ASR Test'].values[0]
           # asr_overall = df['ASR Overall'].values[0]
            linf_evolution = df['Linf Evolution'].values[0]
            linf_test = df['Linf Testing'].values[0]
            l2_evolution = df['L2 Evolution'].values[0]
            l2_test = df['L2 Testing'].values[0]
            acc_old = df['Old Accuracy'].values[0]
            acc_new = df['New Accuracy'].values[0]

            state_log_file = os.path.join(best_folder_path, 'state.log')
            if os.path.isfile(state_log_file):
                with open(state_log_file, 'r') as file:
                    lines = file.readlines()
                    seed_line = next((l for l in lines if 'seed =' in l), None)
                    if seed_line:
                        seed = seed_line.split('=')[1].strip()
                    else:
                        seed = 'Unknown'  # If seed is not found

            results_data.append([scenario,
                model_name,
                best_fitness,
                asr_evolution,
                asr_test,
                linf_evolution,
                linf_test,
                l2_evolution,
                l2_test,
                acc_old,
                acc_new,
                time_min,
                best_folder_path,
                seed,
            ])

        print(f'{scenario} - {1-asr_test}')

    df_results = pd.DataFrame(results_data, columns=columns)

    if not os.path.exists('best_run'):
        os.mkdir('best_run')
    output_file = f'best_run/{model_name}_best_run_details.csv'  # Adjust path if needed
    df_results.to_csv(output_file, index=False)

    print(f"Details of the best runs for {model_name} saved to {output_file}")
