import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
import os

def parse_string_list(s):
    return [int(x) for x in s.strip('[]').split(', ')]

def aggregate_to_blocks(heatmap, block_size):
    max_height, max_width = heatmap.shape
    block_height = max_height // block_size
    block_width = max_width // block_size
    
    aggregated_heatmap = np.zeros((block_height, block_width))
    
    for i in range(block_height):
        for j in range(block_width):
            block = heatmap[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            aggregated_heatmap[i, j] = np.sum(block)
    
    return aggregated_heatmap

if __name__ == "__main__":

    scenarios = ['f1', 'f2', 'f3']
    targets = ['3%', '5%', '10%']
 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    max_width = 32
    max_height = 32
    block_size = 8

    for scenario in scenarios:
        main_folder = f'../runs_final/{scenario}'
        # modelNames = ['ResNet50', 'ResNet101', 'VGG16', 'VGG19']
        modelNames = ['ResNet50','VGG16']
        gen_change = 20

        for m in range(len(modelNames)):
            modelName = modelNames[m]
            best_fitnesses = []
            avg_fitnesses =  []
            best_depths = []
            avg_depths = []
            best_asr_treinos =  []
            avg_asr_treinos = []
            best_pss = []
            avg_pss =  []
            heatmaps_start = []
            heatmapbwh_s = []
            heatmapbxy_s = []
            heatmapxy_bests = []
            heatmaps_middle = []
            heatmaps_end = []
            heatmapsxy_start = []
            heatmapsxy_middle = []
            heatmapsxy_end = []
            avg_areas = []
            best_areas = []
            best_confidences = []
            avg_confidences = []

            for folder_name in os.listdir(main_folder):
                if folder_name.startswith(f'{modelName}'):
                    heatmapbwh = np.zeros((max_width, max_height))
                    heatmapbxy = np.zeros((max_width, max_height))
            
                    run_folder = os.path.join(main_folder, folder_name)
                    ficheiro_adv_l2 = f'{run_folder}/gen_perturbation_size_advs_fit.csv'
                    ficheiro_fit = f'{run_folder}/evolution_{folder_name}.csv'

                    # PERTURBATION SIZE AND ADVERSARIAL SUCCESS RATE
                    l2 = False
                    if l2:
                        name = 'L2'
                        avg_name = 'AVG ' + name
                        best_name = "BEST IND'S " + name
                        ps_max = 32 * np.sqrt(3)
                    else:
                        name = 'L$\infty$'
                        avg_name = 'AVG ' + name
                        best_name = "BEST IND'S " + name
                        ps_max = 1

                    matplotlib.rcParams.update({'font.size': 40})

                    data = pd.read_csv(ficheiro_adv_l2)
                    avg_ps = data['perturbation_size_average']
                    std_ps = data['perturbation_size_std']
                    best_ind_ps = data['best_ind_perturbation_size']

                    avg_advq = data['adv_quantity_average']
                    std_advq = data['adv_quantity_std']
                    best_ind_advq = data['best_ind_advq']
                    std_fit = data['fitness_std']

                    fit_data = pd.read_csv(ficheiro_fit)
                    fit_best = fit_data[' fitness generational best']
                    fit_avg = fit_data[' fitness avg']
                    depth_best = fit_data[' depth generational best']
                    depth_avg = fit_data['depth avg']
                    depth_std = fit_data[' depth std']

                    best_asr_treinos.append(best_ind_advq)
                    avg_asr_treinos.append(avg_advq)

                    best_pss.append(best_ind_ps)
                    avg_pss.append(avg_ps)

                    best_fitnesses.append(fit_best)
                    avg_fitnesses.append(fit_avg)

                    best_depths.append(depth_best)
                    avg_depths.append(depth_avg)

                    patch_folder = f'{run_folder}/logs/perturbation_size_advs_fit_per_gen'
                    ngens = len(avg_ps)
                    gens_heatmap = [0, int( (ngens-1)/2), ngens-1]
                    area_avg = []
                    area_best = []
                    area_std = []

                    best_cy_adv = []
                    avg_cy_adv = []
                    std_cy_adv = []

                    # best_cy_no_adv = []
                    # avg_cy_no_adv = []
                    # std_cy_no_adv = []

                    for g in range(ngens):
                        ficheiro_pathc = f'{patch_folder}/gen_{g}_perturbation_size_advs_fit.csv' 
                        patch_data = pd.read_csv(ficheiro_pathc)
                        values = patch_data['patch'].values
                        arrays_of_int = [parse_string_list(s) for s in values]
                        arrays_of_int = np.array(arrays_of_int) 
                        w_and_h = arrays_of_int[:, 2:] # os dois ultimos numeros do patch
                        b_wh = w_and_h[0]
                        x_and_y = arrays_of_int[:, :2] # os dois primeiros numeros do patch
                        b_xy = x_and_y[0]
                        number_of_ind = len(w_and_h)

                        heatmapbwh[b_wh[1] - 1, b_wh[0] - 1] += 1
                        heatmapbxy[b_xy[1], b_xy[0]] += 1

                        if g in gens_heatmap:

                            heatmapwh = np.zeros((max_height, max_width))
                            heatmapxy = np.zeros((max_height, max_width))

                            for width, height in w_and_h:
                                heatmapwh[height-1, width-1] += 1  # Subtract 1 to adjust to zero-based indexing
                            for x, y in x_and_y:
                                heatmapxy[y, x] += 1  # Subtract 1 to adjust to zero-based indexing
                            
                            if g == gens_heatmap[0]: 
                                heatmaps_start.append(heatmapwh)
                                heatmapsxy_start.append(heatmapxy)
                            elif g == gens_heatmap[1]:
                                heatmaps_middle.append(heatmapwh)
                                heatmapsxy_middle.append(heatmapxy)
                            elif g == gens_heatmap[2]:
                                heatmaps_end.append(heatmapwh)
                                heatmapsxy_end.append(heatmapxy)


                       # folder_confidence_y = f'{run_folder}/tensor_attack/generation_{g}'
                        # best
                        # ind_means = []
                        # for i in range(number_of_ind):
                        #     ficheiro_i = f'{folder_confidence_y}/{i}.csv'
                        #     cy_data = pd.read_csv(ficheiro_i)
                        #     # cy_adv = cy_data['adv']
                        #     cy_act_y = cy_data['act_true_label']
                        #     cy_act_y_adv = []

                        #     #if any(cy_adv):             cy_act_y_adv.append(cy_act_y[cy_adv].values)
                        #     cy_act_y_adv.append(cy_act_y.values)
                        #     # else:
                        #     #    cy_act_y_adv = [1]
                            
                        #     ind_mean = np.mean(cy_act_y_adv)
                        #     ind_means.append(ind_mean)
                        #     if i == 0: best_cy_adv.append(np.mean(cy_act_y_adv))

                        # avg_cy_adv.append(np.mean(ind_means))
                        # std_cy_adv.append(np.std(ind_means))
                        
                        areas = w_and_h[:, 0] * w_and_h[:, 1]
                        area_avg.append(np.mean(areas))
                        area_std.append(np.std(areas))
                        area_best.append(w_and_h[0][0]*w_and_h[0][1])
                    
                   # best_confidences.append(best_cy_adv)
                    avg_confidences.append(avg_cy_adv)
                    avg_areas.append(area_avg)
                    best_areas.append(area_best)
                    heatmapbwh_s.append(heatmapbwh)
                    heatmapbxy_s.append(heatmapbxy)
            ngens = len(avg_ps)
            gens = list(range(0, ngens))

            # PS and ASR
            fig, ax = plt.subplots(1, 1)

            ax.plot(gens[0:-1], np.mean(avg_pss, axis=0)[0:-1], linestyle='-', label=avg_name)
            ax.plot(gens[0:-1], np.mean(best_pss, axis=0)[0:-1], linestyle='-', label=best_name)
            ax.fill_between(gens[0:-1], np.mean(avg_pss, axis=0)[0:-1] - np.std(avg_pss, axis=0)[0:-1], np.mean(avg_pss, axis=0)[0:-1] + np.std(avg_pss, axis=0)[0:-1], alpha=0.3)  # Fill between with std
            ax.fill_between(gens[0:-1], np.mean(best_pss, axis=0)[0:-1] - np.std(best_pss, axis=0)[0:-1], np.mean(best_pss, axis=0)[0:-1] + np.std(best_pss, axis=0)[0:-1], alpha=0.3)  # Fill between with std

            for gen in gens:
                if gen % gen_change == 0:
                    ax.axvline(x=gen, color='gray', linestyle='--')
            
            ax.set_xlabel('Generations')
            ax.set_ylabel(name)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
            ax.set_ylim(0, ps_max + 0.05)
            ax.set_title(f'{name} across generations')
            fig.set_size_inches(12, 8)
            plt.legend(loc='upper right')
            plt.savefig(f'graphs/_{modelName}_{name}_ ' + '.png')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1)
            ax.plot(gens[0:-1], np.mean(avg_asr_treinos, axis=0)[0:-1], linestyle='-', label="AVG ASR")
            ax.plot(gens[0:-1], np.mean(best_asr_treinos, axis=0)[0:-1], linestyle='-', label="BEST IND's ASR")
            ax.fill_between(gens[0:-1], np.mean(avg_asr_treinos, axis=0)[0:-1] - np.std(avg_asr_treinos, axis=0)[0:-1], np.mean(avg_asr_treinos, axis=0)[0:-1] + np.std(avg_asr_treinos, axis=0)[0:-1], alpha=0.3)  # Fill between with std
            ax.fill_between(gens[0:-1], np.mean(best_asr_treinos, axis=0)[0:-1] - np.std(best_asr_treinos, axis=0)[0:-1], np.mean(best_asr_treinos, axis=0)[0:-1] + np.std(best_asr_treinos, axis=0)[0:-1], alpha=0.3)  # Fill between with std

            for gen in gens:
                if gen % gen_change == 0:
                    ax.axvline(x=gen, color='gray', linestyle='--')

            ax.set_xlabel('Generations')
            ax.set_ylabel('ASR')
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
            ax.set_ylim(0, 1.05)
            ax.set_title('Avg ASR across generations')
            fig.set_size_inches(12, 8)
            plt.legend(loc='upper right')
            plt.savefig(f'graphs/_{modelName}_Adv_ ' + '.png')
            plt.close(fig)

            # FITNESS AND DEPTH

            fig, ax = plt.subplots(1, 1)
            ax.plot(gens[0:-1], np.mean(avg_fitnesses, axis=0)[0:-1], linestyle='-', label="AVG")
            ax.plot(gens[0:-1], np.mean(best_fitnesses, axis=0)[0:-1], linestyle='-', label="BEST")
            ax.fill_between(gens[0:-1], np.mean(avg_fitnesses, axis=0)[0:-1] - np.std(avg_fitnesses, axis=0)[0:-1], np.mean(avg_fitnesses, axis=0)[0:-1] + np.std(avg_fitnesses, axis=0)[0:-1], alpha=0.3)  # Fill between with std
            ax.fill_between(gens[0:-1], np.mean(best_fitnesses, axis=0)[0:-1] - np.std(best_fitnesses, axis=0)[0:-1], np.mean(best_fitnesses, axis=0)[0:-1] + np.std(best_fitnesses, axis=0)[0:-1], alpha=0.3)  # Fill between with std

            for gen in gens:
                if gen % gen_change == 0:
                    ax.axvline(x=gen, color='gray', linestyle='--')

            ax.set_xlabel('Generations')
            ax.set_ylabel('Fitness')
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
            ax.set_ylim(0, 1.05)
            ax.set_title('Fitness across generations')
            fig.set_size_inches(12, 8)
            plt.legend(loc='upper right')
            plt.savefig(f'graphs/_{modelName}_Fitness_.png')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1)
            ax.plot(gens[0:-1], np.mean(avg_depths, axis=0)[0:-1], linestyle='-', label="AVG")
            ax.plot(gens[0:-1], np.mean(best_depths, axis=0)[0:-1], linestyle='-', label="BEST")
            ax.fill_between(gens[0:-1], np.mean(avg_depths, axis=0)[0:-1] - np.std(avg_depths, axis=0)[0:-1], np.mean(avg_depths, axis=0)[0:-1] + np.std(avg_depths, axis=0)[0:-1], alpha=0.3)  # Fill between with std
            ax.fill_between(gens[0:-1], np.mean(best_depths, axis=0)[0:-1] - np.std(best_depths, axis=0)[0:-1], np.mean(best_depths, axis=0)[0:-1] + np.std(best_depths, axis=0)[0:-1], alpha=0.3)  # Fill between with std

            for gen in gens:
                if gen % gen_change == 0:
                    ax.axvline(x=gen, color='gray', linestyle='--')

            ax.set_xlabel('Generations')
            ax.set_ylabel('Depth')
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
            ax.set_ylim(0, 8.05)
            ax.set_title('Depth across generations')
            fig.set_size_inches(12, 8)
            plt.legend(loc='lower right')
            plt.savefig(f'graphs/_{modelName}_Depth_.png')
            plt.close(fig)


            # Plot heatmap bests wh -> é a media entre runs, com todos os bests de cada run (um best por geraçao)
            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
            matplotlib.rcParams.update({'font.size': 30})
            aggregated_heatmapbwh = aggregate_to_blocks(np.mean(heatmapbwh_s, axis=0), block_size)
            plt.imshow(aggregated_heatmapbwh, cmap='viridis', aspect='auto', vmin=0, vmax=600)
            plt.colorbar(label='Number of Individuals')
            plt.xlabel('Width', fontsize=35)
            plt.ylabel('Height', fontsize=35)
            block_labels = [f'{i}-{i+block_size-1}' for i in range(0, max_width, block_size)]
            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30) 
            for i in range(aggregated_heatmapbwh.shape[0]):
                for j in range(aggregated_heatmapbwh.shape[1]):
                    plt.text(j, i, f'{aggregated_heatmapbwh[i, j]:.2f}', ha='center', va='center', color='white', fontsize=30)           
            # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
            plt.savefig(f'graphs/{scenario}_{modelName}_WH_best_heatmap.png', bbox_inches='tight')
            plt.close()

            #Plot heatmap bests xy
            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
            aggregated_heatmapbxy = aggregate_to_blocks(np.mean(heatmapbxy_s, axis=0),block_size)
            plt.imshow(aggregated_heatmapbxy, cmap='viridis', aspect='auto', vmin=0, vmax=600)
            plt.colorbar(label='Number of Individuals')
            plt.xlabel('X', fontsize=35)
            plt.ylabel('Y', fontsize=35)
            block_labels = [f'{i}-{i+block_size-1}' for i in range(0, max_width, block_size)]
            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30) 
            for i in range(aggregated_heatmapbxy.shape[0]):
                for j in range(aggregated_heatmapbxy.shape[1]):
                    plt.text(j, i, f'{aggregated_heatmapbxy[i, j]:.2f}', ha='center', va='center', color='white', fontsize=30)           
            # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
            plt.savefig(f'graphs/{scenario}_{modelName}_XY_best_heatmap.png', bbox_inches='tight')
            plt.close()

            # Plot the heatmap wh start
            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
            aggregated_heatmap_start = aggregate_to_blocks(np.mean(heatmaps_start, axis=0), block_size)
            plt.imshow(aggregated_heatmap_start, cmap='viridis', aspect='auto', vmin=0, vmax=len(w_and_h))
            plt.colorbar(label='Number of Individuals')
            plt.xlabel('Width', fontsize=35)
            plt.ylabel('Height', fontsize=35)
            block_labels = [f'{i}-{i+block_size-1}' for i in range(1, max_width+1, block_size)]
            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30) 
            for i in range(aggregated_heatmap_start.shape[0]):
                for j in range(aggregated_heatmap_start.shape[1]):
                    plt.text(j, i, f'{aggregated_heatmap_start[i, j]:.2f}', ha='center', va='center', color='white', fontsize=30)           
            # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
            plt.savefig(f'graphs/{scenario}_{modelName}_WH_heatmap_start.png', bbox_inches='tight')
            plt.close()

            # Plot the heatmap wh midddle
            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
            aggregated_heatmap_middle = aggregate_to_blocks(np.mean(heatmaps_middle, axis=0), block_size)
            plt.imshow(aggregated_heatmap_middle, cmap='viridis', aspect='auto', vmin=0, vmax=len(w_and_h))
            plt.colorbar(label='Number of Individuals')
            plt.xlabel('Width', fontsize=35)
            plt.ylabel('Height', fontsize=35)
            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            for i in range(aggregated_heatmap_middle.shape[0]):
                for j in range(aggregated_heatmap_middle.shape[1]):
                    plt.text(j, i, f'{aggregated_heatmap_middle[i, j]:.2f}', ha='center', va='center', color='white', fontsize=30)    
           #  plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
            plt.savefig(f'graphs/{scenario}_{modelName}_WH_heatmap_middle.png', bbox_inches='tight')
            plt.close()

            # Plot the heatmap wh end
            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
            aggregated_heatmap_end = aggregate_to_blocks(np.mean(heatmaps_end, axis=0), block_size)
            plt.imshow(aggregated_heatmap_end, cmap='viridis', aspect='auto', vmin=0, vmax=len(w_and_h))
            plt.colorbar(label='Number of Individuals')
            plt.xlabel('Width', fontsize=35)
            plt.ylabel('Height', fontsize=35)
            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            for i in range(aggregated_heatmap_end.shape[0]):
                for j in range(aggregated_heatmap_end.shape[1]):
                    plt.text(j, i, f'{aggregated_heatmap_end[i, j]:.2f}', ha='center', va='center', color='white', fontsize=30)   
            # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
            plt.savefig(f'graphs/{scenario}_{modelName}_WH_heatmap_end.png', bbox_inches='tight')
            plt.close()
            
            # xy start
            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
            aggregated_heatmapxy_start = aggregate_to_blocks(np.mean(heatmapsxy_start, axis=0), block_size)
            plt.imshow(aggregated_heatmapxy_start, cmap='viridis', aspect='auto', vmin=0, vmax=len(x_and_y))
            plt.colorbar(label='Number of Individuals')
            plt.xlabel('X', fontsize=35)
            plt.ylabel('Y', fontsize=35)
            block_labels = [f'{i}-{i+block_size-1}' for i in range(0, max_width, block_size)]
            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)  
            for i in range(aggregated_heatmapxy_start.shape[0]):
                for j in range(aggregated_heatmapxy_start.shape[1]):
                    plt.text(j, i, f'{aggregated_heatmapxy_start[i, j]:.2f}', ha='center', va='center', color='white', fontsize=30)           
           # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
            plt.savefig(f'graphs/{scenario}_{modelName}_XY_heatmap_start.png', bbox_inches='tight')
            plt.close()

            # xy middle
            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
            aggregated_heatmapxy_middle = aggregate_to_blocks(np.mean(heatmapsxy_middle, axis=0), block_size)
            plt.imshow(aggregated_heatmapxy_middle, cmap='viridis', aspect='auto', vmin=0, vmax=len(x_and_y))
            plt.colorbar(label='Number of Individuals')
            plt.title('Heatmap of X, Y')
            plt.xlabel('X', fontsize=35)
            plt.ylabel('Y', fontsize=35)
            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)  
            for i in range(aggregated_heatmapxy_middle.shape[0]):
                for j in range(aggregated_heatmapxy_middle.shape[1]):
                    plt.text(j, i, f'{aggregated_heatmapxy_middle[i, j]:.2f}', ha='center', va='center', color='white', fontsize=30)     
           # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
            plt.savefig(f'graphs/{scenario}_{modelName}_XY_heatmap_middle.png', bbox_inches='tight')
            plt.close()

            # xy end
            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
            aggregated_heatmapxy_end = aggregate_to_blocks(np.mean(heatmapsxy_end, axis=0), block_size)
            plt.imshow(aggregated_heatmapxy_end, cmap='viridis', aspect='auto', vmin=0, vmax=len(x_and_y))
            plt.colorbar(label='Number of Individuals')
            plt.title('Heatmap of X, Y')
            plt.xlabel('X', fontsize=35)
            plt.ylabel('Y', fontsize=35)
            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30)
            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=30) 
            for i in range(aggregated_heatmapxy_end.shape[0]):
                for j in range(aggregated_heatmapxy_end.shape[1]):
                    plt.text(j, i, f'{aggregated_heatmapxy_end[i, j]:.2f}', ha='center', va='center', color='white', fontsize=30)   
           # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
            plt.savefig(f'graphs/{scenario}_{modelName}_XY_heatmap_end.png', bbox_inches='tight')
            plt.close()
        
            # AREA
            matplotlib.rcParams.update({'font.size': 25})

            fig, ax = plt.subplots(1, 1)
            ax.plot(gens[0:-1], np.mean(avg_areas, axis=0)[0:-1], linestyle='-', label="AVG Area")
            ax.plot(gens[0:-1], np.mean(best_areas, axis=0)[0:-1], linestyle='-', label="BEST IND's Area")
            ax.fill_between(gens[0:-1], np.mean(avg_areas, axis=0)[0:-1] - np.std(avg_areas, axis=0)[0:-1], np.mean(avg_areas, axis=0)[0:-1] + np.std(avg_areas, axis=0)[0:-1], alpha=0.3)  # Fill between with std
            ax.fill_between(gens[0:-1], np.mean(best_areas, axis=0)[0:-1] - np.std(best_areas, axis=0)[0:-1], np.mean(best_areas, axis=0)[0:-1] + np.std(best_areas, axis=0)[0:-1], alpha=0.3)  # Fill between with std

            for gen in gens:
                if gen % gen_change == 0:
                    ax.axvline(x=gen, color='gray', linestyle='--')

            ax.set_xlabel('Generations')
            ax.set_ylabel('Area')
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
            ax.set_ylim(0, 1025)
            ax.set_title('Patch area across generations')
            fig.set_size_inches(12, 8)
            plt.legend(loc='lower right')
            plt.savefig(f'graphs/{scenario}_{modelName}_Patch_area_.png', bbox_inches='tight')
            plt.close(fig)
            
            # CONFIDENCE IN TRUE LABEL
                
            # fig, ax = plt.subplots(1, 1)
            # ax.plot(gens[0:-1], np.mean(avg_confidences, axis=0)[0:-1], linestyle='-', label="AVG CY")
            # ax.plot(gens[0:-1], np.mean(best_confidences, axis=0)[0:-1], linestyle='-', label="BEST IND's CY")
            # ax.fill_between(gens[0:-1], np.mean(avg_confidences, axis=0)[0:-1] - np.std(avg_confidences, axis=0)[0:-1], np.mean(avg_confidences, axis=0)[0:-1] + np.std(avg_confidences, axis=0)[0:-1], alpha=0.3)  # Fill between with std
            # ax.fill_between(gens[0:-1], np.mean(best_confidences, axis=0)[0:-1] - np.std(best_confidences, axis=0)[0:-1], np.mean(best_confidences, axis=0)[0:-1] + np.std(best_confidences, axis=0)[0:-1], alpha=0.3)  # Fill between with std

            # for gen in gens:
            #     if gen % gen_change == 0:
            #         ax.axvline(x=gen, color='gray', linestyle='--')

            # ax.set_xlabel('Generations')
            # ax.set_ylabel('CY')
            # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
            # ax.set_ylim(0, 1.05)
            # ax.set_title('Confidence in true label across generations')
            # fig.set_size_inches(12, 8)
            # plt.legend(loc='lower right')
            # plt.savefig(f'{main_folder}/_{modelName}_CY_.png')
            # plt.close(fig)

