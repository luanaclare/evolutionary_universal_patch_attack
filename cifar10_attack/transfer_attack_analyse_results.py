import os
import pandas as pd
import numpy as np

scenarios = ['f1','f2', 'f3']
modelNames1 = ['VGG16']
modelNames2 = ['ResNet50', 'ResNet101', 'VGG16', 'VGG19']
columns = [
    'Model', 'ASR Evolution Mean', 'ASR Evolution STD', 'ASR Test Mean', 'ASR Test STD', 'ASR Overall Mean', 'ASR Overall STD',
    'Linf Evolution Mean', 'Linf Evolution STD', 'Linf Test Mean', 'Linf Test STD', 'L2 Evolution Mean', 'L2 Evolution STD',
    'L2 Test Mean', 'L2 Test STD', 'ACC Old', 'ACC New Mean', 'ACC New STD'
]

for scenario in scenarios:
    main_folder = f'runs_final/{scenario}'

    for original_modelName in modelNames1:
        results_data = []

        for attack_modelName in modelNames2:
            if original_modelName != attack_modelName and original_modelName == 'VGG16':
                print('Model under attack ... ', attack_modelName)

                asr_treinos = []
                asr_testes = []
                asr_overalls = []
                linf_treinos = []
                linf_testes = []
                l2_treinos = []
                l2_testes = []
                acc_antiga = None
                acc_novas = []

                for folder_name in os.listdir(main_folder):
                    if folder_name.startswith(original_modelName):
                        folder_path = os.path.join(main_folder, folder_name)
                        if os.path.isdir(folder_path):
                            test_info_file = f'{folder_path}/{attack_modelName}_test_info.csv'
                            df = pd.read_csv(test_info_file)
                            acc_antiga = (df['Old Accuracy'].values)[0]
                            acc_novas.append((df['New Accuracy'].values)[0])
                            linf_treinos.append((df['Linf Evolution'].values)[0])
                            linf_testes.append((df['Linf Testing'].values)[0])
                            l2_treinos.append((df['L2 Evolution'].values)[0])
                            l2_testes.append((df['L2 Testing'].values)[0])
                            asr_testes.append((df['ASR Test'].values)[0])
                            asr_treinos.append((df['ASR Evolution'].values)[0])
                            asr_overalls.append((df['ASR Overall'].values)[0])

                # Calculate mean and std
                def calculate_mean_std(data):
                    return np.mean(data), np.std(data)

                asr_treino_mean, asr_treino_std = calculate_mean_std(asr_treinos)          
                linf_treino_mean, linf_treino_std = calculate_mean_std(linf_treinos)         
                l2_treino_mean, l2_treino_std = calculate_mean_std(l2_treinos)             
                asr_teste_mean, asr_teste_std = calculate_mean_std(asr_testes)            
                linf_teste_mean, linf_teste_std = calculate_mean_std(linf_testes)           
                l2_teste_mean, l2_teste_std = calculate_mean_std(l2_testes)              
                acc_nova_mean, acc_nova_std = calculate_mean_std(acc_novas)            
                asr_overall_mean, asr_overall_std = calculate_mean_std(asr_overalls)          

                results_data.append([
                    attack_modelName,
                    asr_treino_mean, asr_treino_std,
                    asr_teste_mean, asr_teste_std,
                    asr_overall_mean, asr_overall_std,
                    linf_treino_mean, linf_treino_std,
                    linf_teste_mean, linf_teste_std,
                    l2_treino_mean, l2_treino_std,
                    l2_teste_mean, l2_teste_std,
                    acc_antiga, acc_nova_mean, acc_nova_std
                ])

        # Create a DataFrame from the collected data
        df = pd.DataFrame(results_data, columns=columns)

        # Save the DataFrame to a CSV file
        output_file = f'{main_folder}/transfer_attack_dataset_{original_modelName}.csv'
        df.to_csv(output_file, index=False)

        print(f"Dataset saved to {output_file}")
