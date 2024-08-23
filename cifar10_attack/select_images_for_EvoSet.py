import pandas as pd
from setup_cifar import CIFAR
import numpy as np
import csv
import torchvision
# 'regular', 'distilled', 'VGG16', 'ResNet50', 
modelNames = ['VGG19', 'ResNet101']

total_evo_subset_size = 1000
mini_evo_subset_size = 200
per_class = int(total_evo_subset_size/10)
qnt_subsets = int(total_evo_subset_size/mini_evo_subset_size)

seed = 2020

np.random.seed(seed)
for modelName in modelNames:
    # Get images   
    images_df = pd.read_csv(f'correctly_classified_images/{modelName}_correctly_classified_images.csv')
    images_idx = images_df['image id']
    images_labels = images_df['true label']
    
    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)
    y_test = testset.targets
    y_test = np.array(y_test)
    
    class_counts = images_labels.value_counts()
    random_indices = []

    for label in class_counts.index:
        indices = images_idx[images_labels == label].tolist()
        selected_indices = np.random.choice(indices, per_class, replace=False)
        random_indices.extend(selected_indices)
    
    random_indices = np.array(random_indices)
    np.random.shuffle(random_indices)
    csv_subset = f'selected_images_EvoSet/{modelName}_cifar_attack_subset_{seed}_{qnt_subsets}x{mini_evo_subset_size}.csv'

    with open(csv_subset, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["index", "label"])
        for inx in random_indices:
            csv_writer.writerow([inx, y_test[inx]])