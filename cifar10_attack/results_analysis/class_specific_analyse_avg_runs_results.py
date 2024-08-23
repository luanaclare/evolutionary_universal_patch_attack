import os
import pandas as pd
import numpy as np

scenarios = ['f1', 'f2','f3']
columns = [
            'Model', 'Best fitness final', 'STD Best fitness final', 'ASR Evolution', 'STD ARS Evolution', 'ASR Test', 'STD ARS Test', 
            'ASR Overall', 'STD ASR Overall','Linf Evolution', 'STD Linf Evolution', 'Linf Test', 'STD Linf Test', 'L2 Evolution', 'STD L2 Evolution',
            'L2 Test', 'STD L2 Test',
            'ACC Old', 'ACC New', 'ACC New STD', 'tempo (min)'
        ]
#modelNames = ['ResNet50', 'ResNet101', 'VGG16', 'VGG19']
modelNames = ['VGG16']

for scenario in scenarios:

    for class_specific in range(10):
        for scenario in scenarios:
            main_folder = f'../runs_final/{scenario}_class_{class_specific}'
            results_data = []

            for modelName in modelNames:
                times = []
                best_fitnesses = []
                asr_treinos = []
                asr_testes = []
                asr_overalls = []
                linf_treinos = []
                linf_testes = []
                l2_treinos = []
                l2_testes = []
                acc_novas = []

                for folder_name in os.listdir(main_folder):
                    if folder_name.startswith(modelName):
                        folder_path = os.path.join(main_folder, folder_name)
                        if os.path.isdir(folder_path):
                            logs_folder_path = os.path.join(folder_path, 'logs')
                            if os.path.isdir(logs_folder_path):
                                for file_name in os.listdir(logs_folder_path):
                                    if file_name.startswith('timings') and file_name.endswith('.csv'):
                                        file_path = os.path.join(logs_folder_path, file_name)
                                        df = pd.read_csv(file_path)
                                        time_value = df[' total'].values
                                        times.append(time_value[0]/60)
                            best_info_file = f'{folder_path}/best_info.csv'
                            df = pd.read_csv(best_info_file)
                            best_fitnesses.append((df['fitness'].values)[0])
                            asr_treinos.append((df['adv_quantity'].values)[0])

                            test_info_file = f'{folder_path}/test_info.csv'
                            df = pd.read_csv(test_info_file)
                            acc_antiga = (df['Old Accuracy (in class)'].values)[0]
                            acc_novas.append((df['New Accuracy (in class)'].values)[0])
                            linf_treinos.append((df['Linf Evolution'].values)[0])
                            linf_testes.append((df['Linf Testing'].values)[0])
                            l2_treinos.append((df['L2 Evolution'].values)[0])
                            l2_testes.append((df['L2 Testing'].values)[0])
                            asr_testes.append((df['ASR Test'].values)[0])
                            asr_overalls.append((df['ASR Overall'].values)[0])


                best_fitness_final = np.mean(best_fitnesses)  
                asr_treino = np.mean(asr_treinos)          
                linf_treino = np.mean(linf_treinos)         
                l2_treino = np.mean(l2_treinos)             
                asr_teste = np.mean(asr_testes)            
                linf_teste = np.mean(linf_testes)           
                l2_teste = np.mean(l2_testes)              
                tempo_min = np.mean(times)
                acc_nova = np.mean(acc_novas) 
                acc_nova_std = np.std(acc_novas)           
                asr_overall = np.mean(asr_overalls)          

                results_data.append([
                    modelName,
                    best_fitness_final, np.std(best_fitnesses), asr_treino, np.std(asr_treinos), asr_teste, np.std(asr_testes), 
                    asr_overall, np.std(asr_overalls), linf_treino, np.std(linf_treinos), linf_teste, np.std(linf_testes), l2_treino, np.std(l2_treinos), l2_teste, np.std(l2_testes),
                    acc_antiga, acc_nova, acc_nova_std, tempo_min
                ])
                

            # Create a DataFrame from the collected data
            df = pd.DataFrame(results_data, columns=columns)

            # Save the DataFrame to a CSV file
            output_file = f'{main_folder}/dataset.csv'
            df.to_csv(output_file, index=False)

            print(f"Dataset saved to {output_file}")

