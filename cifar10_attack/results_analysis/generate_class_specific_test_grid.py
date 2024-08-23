import os
import pandas as pd
import numpy as np
import tensorflow as tf
from cifar10_attack.class_specific_main import compute_l2_distance, transform_image, compute_linf_distance, apply_perturbation
import matplotlib.pyplot as pyplot
from PIL import Image
from ast import literal_eval
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from vgg import VGG
from resnet import ResNet50, ResNet101
from torch.utils.data import TensorDataset, DataLoader
import csv

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def save_images_for_scenario(scenario, images_dict, output_dir='test_grid_images'):
    """Plot and save images for a given scenario."""
    rows = 10  # Number of rows in the plot
    columns = 20  # Number of columns in the plot
    fig, axes = pyplot.subplots(rows, columns, figsize=(12, 12))
    axes = axes.flatten()
    for img, ax in zip(images_dict.get(scenario, []), axes):
        ax.imshow(img.astype(np.uint8))
        ax.axis('off')
    pyplot.tight_layout()
    output_path = os.path.join(output_dir, f'{scenario}_test_grid.png')
    pyplot.savefig(output_path, bbox_inches='tight')
    pyplot.close()

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scenarios = ['f1', 'f2', 'f3']
    modelNames = ['VGG16']
    modelClasses = [VGG('VGG16')]
    # chose a run for each scenario - class
    folder_paths = {
    ('f1', 0): '../runs_final/f1_class_0/VGG16__run__2024_07_31__19_00_41_868__112882643230261248',
    ('f1', 1): '../runs_final/f1_class_1/VGG16__run__2024_07_31__22_55_35_596__112883566876819456',
    ('f1', 2): '../runs_final/f1_class_2/VGG16__run__2024_08_01__05_02_40_942__112885010329894912',
    ('f1', 3): '../runs_final/f1_class_3/VGG16__run__2024_08_03__02_40_57_485__112895777668136960',
    ('f1', 4): '../runs_final/f1_class_4/VGG16__run__2024_08_03__09_54_13_095__112897481316433920',
    ('f1', 5): '../runs_final/f1_class_5/VGG16__run__2024_08_03__14_55_36_305__112898666417684480',
    ('f1', 6): '../runs_final/f1_class_6/VGG16__run__2024_08_03__23_10_17_064__112900611575906304',
    ('f1', 7): '../runs_final/f1_class_7/VGG16__run__2024_08_04__00_44_21_942__112900981518630912',
    ('f1', 8): '../runs_final/f1_class_8/VGG16__run__2024_08_04__06_18_14_084__112902294345089024',
    ('f1', 9): '../runs_final/f1_class_9/VGG16__run__2024_08_04__12_33_26_128__112903769694404608',
    ('f2', 0): '../runs_final/f2_class_0/VGG16__run__2024_08_02__06_05_07_184__112890918154010624',
    ('f2', 1): '../runs_final/f2_class_1/VGG16__run__2024_08_02__17_33_12_215__112893623809802240',
    ('f2', 2): '../runs_final/f2_class_2/VGG16__run__2024_08_02__23_46_19_796__112895091002310656',
    ('f2', 3): '../runs_final/f2_class_3/VGG16__run__2024_08_03__03_17_37_584__112895921853825024',
    ('f2', 4): '../runs_final/f2_class_4/VGG16__run__2024_08_03__10_34_17_760__112897638908559360',
    ('f2', 5): '../runs_final/f2_class_5/VGG16__run__2024_08_03__12_19_55_808__112898054278873088',
    ('f2', 6): '../runs_final/f2_class_6/VGG16__run__2024_08_03__19_30_45_986__112899748396138496',
    ('f2', 7): '../runs_final/f2_class_7/VGG16__run__2024_08_04__04_28_48_284__112901864048820224',
    ('f2', 8): '../runs_final/f2_class_8/VGG16__run__2024_08_04__08_28_58_467__112902808434573312',
    ('f2', 9): '../runs_final/f2_class_9/VGG16__run__2024_08_04__12_03_28_525__112903651886694400',
    ('f3', 0): '../runs_final/f3_class_0/VGG16__run__2024_08_01__23_46_53_464__112889430898376704',
    ('f3', 1): '../runs_final/f3_class_1/VGG16__run__2024_08_02__12_51_03_558__112892514373337088',
    ('f3', 2): '../runs_final/f3_class_2/VGG16__run__2024_08_02__17_55_01_317__112893709603110912',
    ('f3', 3): '../runs_final/f3_class_3/VGG16__run__2024_08_03__04_29_45_024__112896205456932864',
    ('f3', 4): '../runs_final/f3_class_4/VGG16__run__2024_08_03__09_37_17_640__112897414767575040',
    ('f3', 5): '../runs_final/f3_class_5/VGG16__run__2024_08_03__16_07_37_945__112898949640683520',
    ('f3', 6): '../runs_final/f3_class_6/VGG16__run__2024_08_03__22_06_32_660__112900360939765760',
    ('f3', 7): '../runs_final/f3_class_7/VGG16__run__2024_08_04__02_37_52_737__112901427870892032',
    ('f3', 8): '../runs_final/f3_class_8/VGG16__run__2024_08_04__07_47_33_430__112902645575188480',
    ('f3', 9): '../runs_final/f3_class_9/VGG16__run__2024_08_04__11_35_41_540__112903542639165440',
    }



    rows = 10
    columns = 20

    for m in range(len(modelNames)):
        modelName = modelNames[m]
        modelClass = modelClasses[m]

        classif = modelClass
        checkpoint_file = f'../../pytorch-cifar-master/checkpoint/{modelName}/ckpt.pth'
        classif = classif.to(device)

        if device == 'cuda':
            classif = torch.nn.DataParallel(classif)

        checkpoint = torch.load(checkpoint_file)
        classif.load_state_dict(checkpoint['net'])
        classif.eval()

        for scenario in scenarios:
            pyplot.figure(figsize=(12, 6))
            pyplot.subplots_adjust(hspace=0.2, wspace=0.05)
            images_all = []
            for class_specific in range(0, 10):
                main_folder = f'../runs_final/{scenario}_class_{class_specific}'

                run_folder = folder_paths.get((scenario, class_specific))
                data_best = pd.read_csv(f'{run_folder}/best_info.csv')
                perturbation_size = data_best['perturbation_size'].values
                patch = data_best['perturbation'].iloc[0]
                patch = literal_eval(patch)
                adv_q = data_best['adv_quantity'].values

                # DATA
                # Load data
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                # Load CIFAR-10 test dataset
                testset = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True)
                
                testset_transformed = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform_test)
                

                all_file = pd.read_csv(f'../correctly_classified_images/{modelName}_correctly_classified_images.csv')

                all_indices = all_file['image id']
                all_indices_labels = all_file['true label']
                all_indices = [index for index,label in zip(all_indices.values, all_indices_labels.values) if label == class_specific]
                
                used_file = pd.read_csv(f'../selected_images_EvoSet/{modelName}_cifar_attack_subset_2020_5x200.csv')
                used_indices = used_file['index']
                used_indices_labels = used_file['label']
                used_indices = [index for index,label in zip(used_indices.values, used_indices_labels.values) if label == class_specific]

                unused_indices = [index for index in all_indices if index not in used_indices]        

                subset_test = testset.data[unused_indices]
                subset_evo = testset.data[used_indices]
                targets_array = np.array(testset.targets)
                subset_y_test = targets_array[unused_indices]
                subset_y_evo = targets_array[used_indices]

                # Perturb test subset
                perturbed_images_test = np.array([apply_perturbation(image, patch) for image in subset_test])
                true_labels_test = torch.tensor(subset_y_test, dtype=torch.long)

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
                    if predicted_numbers[jadv] != subset_y_test[jadv]:
                        wrong_classification.append(jadv)

                wrong_classification = np.array(wrong_classification)
                adversarial_images = [perturbed_images_test[i] for i in wrong_classification]
                # Add images to the dictionary for plotting
                images_all.extend(adversarial_images[:20])  # Collect up to 20 images per class      
                            


            ab = 0

            for i, img in enumerate(images_all):
                if ab == 200 : 
                    break
                sub = pyplot.subplot(rows, columns, 1 + i)
                pyplot.axis('off')
                pyplot.imshow(img.astype(np.uint8), vmin=0, vmax=255)
                ab = ab + 1

                # Save the grid image

            pyplot.savefig(f'advs/class_specific{scenario}_{modelName}_test_grid.png', bbox_inches='tight')
            pyplot.close()