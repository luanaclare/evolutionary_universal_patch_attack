import os
import pandas as pd

scenario = 'f3'
modelNames = ['VGG16']

max_asr_test = -float('inf')
best_folder = None
best_seed = None

main_folder = f'../runs_final/{scenario}'

for modelName in modelNames:
    for folder_name in os.listdir(main_folder):
        if folder_name.startswith(f'{modelName}'):
            folder_path = os.path.join(main_folder, folder_name)
            if os.path.isdir(folder_path):
                test_info_file = os.path.join(folder_path, 'test_info.csv')
                state_log_file = os.path.join(folder_path, 'state.log')
                
                if os.path.isfile(test_info_file) and os.path.isfile(state_log_file):
                    # Read test_info file to get ASR Test value
                    test_df = pd.read_csv(test_info_file)
                    asr_test = test_df['ASR Test'].values[0]

                    # Extract seed from state.log file
                    with open(state_log_file, 'r') as file:
                        lines = file.readlines()
                        for line in lines:
                            # Look for the line that contains the seed value
                            if line.startswith('seed ='):
                                seed_value = line.split('=')[1].strip()
                                break
                        else:
                            seed = 'Unknown'  # If seed is not found

                    if asr_test > max_asr_test:
                        max_asr_test = asr_test
                        best_folder = folder_path
                        best_seed = seed_value

if best_folder and best_seed:
    print(asr_test, best_folder, best_seed)
