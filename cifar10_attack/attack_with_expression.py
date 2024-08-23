# attack test set with expression
import pandas as pd
import os
import numpy as np
import ast
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from vgg import VGG
from resnet import ResNet50, ResNet101
from torch.utils.data import TensorDataset, DataLoader
from main import compute_l2_distance, transform_image, compute_linf_distance, apply_perturbation
import matplotlib.pyplot as plt
import csv
absolute_path = os.path.abspath('../')
sys.path.append(absolute_path)

from tensorgp.engine import *

def create_engine(resolution):
    # resolution = [resolution, resolution, 3] #jncor could be better #[4096, 4096,3] # 4096x4096
    #fset = {'abs', 'add', 'and', 'cos', 'div', 'exp', 'frac', 'if', 'log',
     #   'max', 'mdist', 'min', 'mult', 'neg', 'or', 'pow', 'sin', 'sqrt',  'sub', 'tan', 'xor','lerp', 'sstepp', 'sign', 'clip', 'mod', 'len'}

    # setup Engine for coin
    new_engine = Engine(target_dims=resolution,
                    codomain=[-1, 1],
                    domain=[-1, 1],
                    effective_dims=2,
                    do_final_transform=True,
                    final_transform=[-255, 255],
                    tf_type=np.int8,
                    # polar_coordinates=True,
                    # do_polar_mask=True,
                    # polar_mask_value="scalar(-1.0, -1.0, -1.0)",
                    mkdirrun=False)

    return new_engine, resolution

def get_patch_from_exp(exp, coord):
    x = coord[0]
    y = coord[1]
    w = coord[2]
    h = coord[3]

    new_engine, re = create_engine([w,h,3])
    _,tree = str_to_tree(exp, new_engine.get_terminal_set().set)
    tensor = new_engine.domain_mapping(tree.get_tensor(new_engine))

    tensor_cpu = tensor.cpu()
    patch = np.zeros((32,32,3))
    patch[x:x+w, y:y+h] = tensor_cpu[0:w, 0:h]

    return patch

def attack_set(perturbation, set, sety, classif):
    perturbed_images_test = np.array([apply_perturbation(image, perturbation) for image in set])
    true_labels_test = torch.tensor(sety, dtype=torch.long)

    mean = np.mean(perturbed_images_test, axis=(0, 1, 2))   # Normalize to [0, 1]
    std = np.std(perturbed_images_test, axis=(0, 1, 2))  

    perturbed_images_transformed = torch.stack([transform_image(image,mean,std) for image in perturbed_images_test])
    perturbed_images_tensor = torch.tensor(perturbed_images_transformed)
    perturbed_transformed_dataset = TensorDataset(perturbed_images_tensor, true_labels_test)
    testloader_perturbed = DataLoader(perturbed_transformed_dataset, batch_size=100, shuffle=False, num_workers=0)

    # Predict with perturbed scaled images (TEST)
    predictions = []
    predicted_numbers = []
    activations = []
    with torch.no_grad():
        for data_ in testloader_perturbed:    
            scaled_perturbed_images, true_labels = data_[0].to(device), data_[1].to(device)
            scaled_perturbed_images = scaled_perturbed_images.float()
        # Predict with perturbed scaled images
            predictions_aux = classif(scaled_perturbed_images)
            predictions_aux = predictions_aux.tolist()
            max_logits = np.max(predictions_aux, axis=1, keepdims=True)
            shifted_logits = predictions_aux - max_logits
            exp_logits = np.exp(shifted_logits)
            predictions_aux = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            predicted_numbers_aux = np.argmax(predictions_aux, axis=1)
            activations_aux = np.max(predictions_aux, axis=1)

            predictions.extend(predictions_aux)
            activations.extend(activations_aux)
            predicted_numbers.extend(predicted_numbers_aux)


    wrong_classification = []
    for jadv in range(len(predicted_numbers)):
        if predicted_numbers[jadv] != sety[jadv]:
            wrong_classification.append(jadv)

    wrong_classification = np.array(wrong_classification)
    perturbation_test_l2 = compute_l2_distance(set, perturbed_images_test)
    perturbation_test_linf = compute_linf_distance(set, perturbed_images_test)

    asr_test = len(wrong_classification) / len(set)

    adversarial_images = [perturbed_images_test[i] for i in wrong_classification]
    num_images = len(adversarial_images)
    rows = 10
    columns = 20

    ab = 0
    plt.figure(figsize=(20, 10))  # Set figure size for better visualization
    for i, img in enumerate(adversarial_images):
        if ab == 200:
            break
        sub = plt.subplot(rows, columns, 1 + i)
        plt.axis('off')
        plt.imshow(img.astype(np.uint8), vmin=0, vmax=255)
        ab += 1

    # Save the grid image
    if not os.path.exists('results_analysis/advs'):
        os.mkdir('results_analysis/advs')
        
    plt.savefig(f'results_analysis/advs/adversarial_examples_from_exp.png', bbox_inches='tight')
    plt.close()

    return asr_test, np.mean(perturbation_test_l2), np.mean(perturbation_test_linf)

def perform_attack(perturbation, modelName, classif, read_exp=True):

    data = {}
    data['Model'] = modelName
    all_indices = pd.read_csv(f'correctly_classified_images/{modelName}_correctly_classified_images.csv')['image id']
    used_indices = pd.read_csv(f'selected_images_EvoSet/{modelName}_cifar_attack_subset_2020_5x200.csv')['index']
    unused_indices = [index for index in all_indices.values if index not in used_indices.values]        

    subset_test = testset.data[unused_indices]
    subset_evo = testset.data[used_indices]
    targets_array = np.array(testset.targets)
    subset_y_test = targets_array[unused_indices]
    subset_y_evo = targets_array[used_indices]

    asr_test, perturbation_test_l2, perturbation_test_linf = attack_set(perturbation, subset_test, subset_y_test, classif)
    data['ASR Test'] = asr_test
    data['L2 Test'] = perturbation_test_l2
    data['Linf Test'] = perturbation_test_linf

    if read_exp:
        asr_evo, perturbation_evo_l2, perturbation_evo_linf = attack_set(perturbation, subset_evo, subset_y_evo, classif)
        data['ASR Evo'] = asr_evo
        data['L2 Evo'] = perturbation_evo_l2
        data['Linf evo'] = perturbation_evo_linf

    return data



insert_exp = False
read_exp = not insert_exp
device = 'cuda' if torch.cuda.is_available() else 'cpu'
modelName = 'VGG16'
classif = VGG('VGG16')

checkpoint_file = f'../pytorch-cifar-master/checkpoint/{modelName}/ckpt.pth'
classif = classif.to(device)

if device == 'cuda':
    classif = torch.nn.DataParallel(classif)

checkpoint = torch.load(checkpoint_file)
classif.load_state_dict(checkpoint['net'])
classif.eval()

if insert_exp:
    exp = input('Expression: ')
    w = int(input('W: '))
    h = int(input('H: '))
    x = int(input('X: '))
    y = int(input('Y: '))
    coord = [x,y,w,h]
    perturbation = get_patch_from_exp(exp, coord)   
    
    attack_info = perform_attack(perturbation, modelName, classif, read_exp)

else:
    ind_number = 0
    ind_gen = [600]
    scenario = 'f3'

    for ig in ind_gen:
        ind_gen_str = str(ig)
        formatted_str = ind_gen_str.zfill(5)
        main_folder = f'runs_final/{scenario}'

        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
        testset = torchvision.datasets.CIFAR10(
                            root='./data', train=False, download=True)
                        
        testset_transformed = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        for folder_name in os.listdir(main_folder):
            if folder_name.startswith(f'{modelName}'):
                run_folder = os.path.join(main_folder, folder_name)
                file = f'{run_folder}/logs/generations/gen{formatted_str}.csv'
                data = pd.read_csv(file)
                data[' perturbation'] = data[' perturbation'].apply(ast.literal_eval)
                perturbations = data[' perturbation'].values
                perturbation = perturbations[ind_number]

                # exp = data[' expression'][ind_number]
                
                # file2 = f'{run_folder}/logs/perturbation_size_advs_fit_per_gen/gen_{ind_gen}_perturbation_size_advs_fit.csv'
                # data2 = pd.read_csv(file2)
                # data2['patch'] = data2['patch'].apply(ast.literal_eval)
                # coord = data2['patch'][ind_number] #xywh
                
                # p = get_patch_from_exp(exp, coord)

                # comparison = p == perturbation
                # print(np.all(comparison))

                attack_info = perform_attack(perturbation, modelName, classif, read_exp)

                csv_file_path = f"{main_folder}/gen{ig}_ind{ind_number}_attack_info.csv"
                
                if os.path.exists(csv_file_path):
                    mode = 'a'
                else:
                    mode = 'w'

                with open(csv_file_path, mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=attack_info.keys())

                    if mode == 'w':
                        writer.writeheader()

                    writer.writerow(attack_info)

                print(f"CSV file saved to {csv_file_path}")

