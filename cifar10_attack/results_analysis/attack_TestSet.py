import os
import pandas as pd
import numpy as np
import tensorflow as tf
from  cifar10_attack.main import compute_l2_distance, transform_image, compute_linf_distance, apply_perturbation
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

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scenarios = ['f1', 'f2', 'f3']
    mamama=0
    for scenario in scenarios:
        main_folder = f'runs_final/{scenario}'
        modelNames = ['VGG16', 'ResNet50']
        modelClasses = [VGG('VGG16'), ResNet50()]
        
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

            for folder_name in os.listdir(main_folder):
                if folder_name.startswith(modelName):
                    run_folder = os.path.join(main_folder, folder_name)
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
                    

                    all_indices = pd.read_csv(f'../correctly_classified_images/{modelName}_correctly_classified_images.csv')['image id']
                    used_indices = pd.read_csv(f'../selected_images_EvoSet/{modelName}_cifar_attack_subset_2020_5x200.csv')['index']
                    unused_indices = [index for index in all_indices.values if index not in used_indices.values]        

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
                    
                    # Perturb evo subset
                    perturbed_images_evo = np.array([apply_perturbation(image, patch) for image in subset_evo])
                    true_labels_evo = torch.tensor(subset_y_evo, dtype=torch.long)

                    mean = np.mean(perturbed_images_evo, axis=(0, 1, 2))   # Normalize to [0, 1]
                    std = np.std(perturbed_images_evo, axis=(0, 1, 2))  

                    perturbed_images_transformed_evo = torch.stack([transform_image(image,mean,std) for image in perturbed_images_evo])
                    perturbed_images_tensor_evo = torch.tensor(perturbed_images_transformed_evo)
                    perturbed_transformed_dataset_evo = TensorDataset(perturbed_images_tensor_evo, true_labels_evo)
                    testloader_perturbed_evo = DataLoader(perturbed_transformed_dataset_evo, batch_size=len(used_indices), shuffle=False, num_workers=0)
                    
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

                    # Predict with perturbed scaled images (EVO)
                    with torch.no_grad():
                        for data_ in testloader_perturbed_evo:    
                            scaled_perturbed_images_evo, true_labels_evo = data_[0].to(device), data_[1].to(device)
                            scaled_perturbed_images_evo = scaled_perturbed_images_evo.float()
                        # Predict with perturbed scaled images
                            predictions_evo = classif(scaled_perturbed_images_evo)
                            predictions_evo = predictions_evo.tolist()
                            max_logits = np.max(predictions_evo, axis=1, keepdims=True)
                            shifted_logits = predictions_evo - max_logits
                            exp_logits = np.exp(shifted_logits)
                            predictions_evo = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                            predicted_numbers_evo = np.argmax(predictions_evo, axis=1)
                            activations_evo = np.max(predictions_evo, axis=1)

                    wrong_classification_evo = []
                    for jadv in range(len(predicted_numbers_evo)):
                        if predicted_numbers_evo[jadv] != subset_y_evo[jadv]:
                            wrong_classification_evo.append(jadv)

                    wrong_classification_evo = np.array(wrong_classification_evo)

                    perturbation_test_l2 = compute_l2_distance(subset_test, perturbed_images_test)
                    perturbation_test_linf = compute_linf_distance(subset_test, perturbed_images_test)

                    perturbation_evo_l2 = compute_l2_distance(subset_evo, perturbed_images_evo)
                    perturbation_evo_linf = compute_linf_distance(subset_evo, perturbed_images_evo)

                    acc = len(all_indices)/len(testset)
                    new_acc = (len(all_indices) - len(wrong_classification) - len(wrong_classification_evo))/len(testset)

                    # with open(f"{run_folder}/test_info.txt", "w") as f:
                    #     f.write(f"{len(unused_indices)} samples not used in evolution, {len(used_indices)} samples used in evolution.\n")
                    #     f.write("\nASR in samples not used in evolution: {:.2%}\n".format(len(wrong_classification) / len(subset_test)))
                    #     f.write("ASR in samples used in last subset evolution: {:.2%}\n".format(adv_q[0]))
                    #     f.write("ASR in ALL samples used evolution: {:.2%}\n".format(len(wrong_classification_evo)/len(subset_evo)))
                    #     f.write("\nPerturbation size in evolution subset: {}\n".format(perturbation_size[0]))
                        
                    #     f.write("\nL2:")
                    #     f.write("\nTesting: {}\n".format(np.mean(perturbation_test_l2)))
                    #     f.write("Evolution: {}\n".format(np.mean(perturbation_evo_l2)))

                    #     f.write("\nLinf:")
                    #     f.write("\nTesting: {}\n".format(np.mean(perturbation_test_linf)))
                    #     f.write("Evolution: {}\n".format(np.mean(perturbation_evo_linf)))

                    #     f.write(f'\nOld accuracy: {acc}')
                    #     f.write(f'\nNew accuracy: {new_acc}')
                    #     f.write(f'\nNew accuracy: {new_acc}')

                    header = ['Unused Samples Count', 'Used Samples Count', 'ASR Test', 'ASR Evolution', 'ASR Overall',
                    'Perturbation Size Evolution', 'L2 Testing', 'L2 Evolution', 'Linf Testing', 'Linf Evolution',
                    'Old Accuracy', 'New Accuracy']

                    data = [
                        [len(unused_indices), len(used_indices), len(wrong_classification) / len(subset_test),
                        len(wrong_classification_evo) / len(subset_evo), (len(wrong_classification_evo) + len(wrong_classification))/ len(all_indices), perturbation_size[0],
                        np.mean(perturbation_test_l2), np.mean(perturbation_evo_l2),
                        np.mean(perturbation_test_linf), np.mean(perturbation_evo_linf),
                        acc, new_acc]
                    ]

                    # Define the file path for the CSV file
                    csv_file_path = f"{run_folder}/test_info.csv"

                    # Write data to the CSV file
                    with open(csv_file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        
                        # Write the header
                        writer.writerow(header)
                        
                        # Write the data
                        writer.writerows(data)

                    print(f"CSV file saved to {csv_file_path}")


                    pyplot.figure(figsize=(12, 6))
                    pyplot.subplots_adjust(hspace=0.05, wspace=0.05)
                    adversarial_images = [perturbed_images_test[i] for i in wrong_classification]

                    rows = 10
                    columns = 20
                    ab =0
                    for i, img in enumerate(adversarial_images):
                        if ab == 200 : 
                            break
                        sub = pyplot.subplot(rows, columns, 1 + i)
                        pyplot.axis('off')
                        pyplot.imshow(img.astype(np.uint8), vmin=0, vmax=255)
                        ab = ab + 1

                    # Save the grid image
                    if not os.path.exists('advs'):
                        os.mkdir('advs')

                    pyplot.savefig(f'advs/{scenario}_{modelName}_test_grid_{folder_name}.png', bbox_inches='tight')
                    pyplot.close()
                    