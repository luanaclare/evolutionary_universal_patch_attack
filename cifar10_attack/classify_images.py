import numpy as np
import csv 
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from vgg import VGG
from resnet import ResNet50, ResNet101

if __name__ == "__main__":
    use_tfkeras = True

    # Define transformation for test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # Load the trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelNames = ['VGG16', 'ResNet50', 'VGG19', 'ResNet101']
    models = [VGG('VGG16'), ResNet50(), VGG('VGG19'), ResNet101()]
    np.random.seed(2020)
    
    for mod in range(len(models)):
        net = models[mod]
        net = net.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
        
        checkpoint_file = f'../pytorch-cifar-master/checkpoint/{modelNames[mod]}/ckpt.pth'
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        net.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0

        # export to csv (save images that are correctly classified)
        file_images = f'correctly_classified_images/{modelNames[mod]}_correctly_classified_images.csv'
        f_img = open(file_images, 'w')
        writer_img = csv.writer(f_img)
        header = ['image id', 'true label', 'confidence', 'prediction']
        writer_img.writerow(header)
        nbatch = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                predictions = net(images)
                confidences, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_pred = []
                max_logits = np.max(predictions.tolist(), axis=1, keepdims=True)
                shifted_logits = predictions.tolist() - max_logits
                exp_logits = np.exp(shifted_logits)
                predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                for i in range(len(predictions)):
                    image_id = i + nbatch*100
                    confidence = np.max(predictions[i])
                    label = labels[i].item()
                    y_pred.append(predicted[i].item())

                    if predicted[i].item() == label:
                        prediction_values = predictions[i].tolist()
                        writer_img.writerow([image_id, label, confidence, prediction_values])
                nbatch += 1

        f_img.close()
        accuracy = correct / total
        print("Accuracy: ", accuracy)

        # Save accuracy to a text file
        accuracy_file_path = 'correctly_classified_images/accuracy.txt'
        with open(accuracy_file_path, 'a') as accuracy_file:
            accuracy_file.write(f'{modelNames[mod]} - Accuracy: {accuracy}\n')

        print(f'Accuracy saved to: {accuracy_file_path}')
