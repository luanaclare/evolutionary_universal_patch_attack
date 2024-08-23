import os
import pandas as pd

# Define the scenario and model names
scenarios = ['f1', 'f2', 'f3']
modelNames = ['VGG16']

for scenario in scenarios:
    print(scenario)
    # Path to the main folder
    for lab in range(10):
        max_asr_test = -float('inf')
        chosen_acc_new = 0
        best_folder = None
        best_seed = None

        main_folder = f'../runs_final/{scenario}_class_{lab}'

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
                            acc_new = test_df['New Accuracy (in class)'].values[0]

                            # Extract seed from state.log file
                            seed_value = 'Unknown'
                            with open(state_log_file, 'r') as file:
                                lines = file.readlines()
                                for line in lines:
                                    # Look for the line that contains the seed value
                                    if line.startswith('seed ='):
                                        seed_value = line.split('=')[1].strip()
                                        break

                            # Check if this is the highest ASR Test value
                            if asr_test > max_asr_test:
                                max_asr_test = asr_test
                                chosen_acc_new = acc_new
                                best_folder = folder_path
                                best_seed = seed_value

        # Print the results for the current scenario and class
        if best_folder and best_seed:
            print(best_seed)

