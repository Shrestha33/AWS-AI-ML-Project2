import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
from workspace_utils import active_session


def data_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test' 
    
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = valid_test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    return trainloader, validloader, testloader, train_data.class_to_idx


def build_model(arch, hidden_units, class_to_idx):
    if arch == 'vgg16':
        input_size = 25088
    elif arch == 'alexnet':
        input_size = 9216
    elif arch == 'densenet121':
        input_size = 1024
    else:
        input_size = 25088
        
    model = getattr(models, arch)(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    
    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(p = 0.2),
                                 nn.Linear(hidden_units, 400),
                                 nn.ReLU(),
                                 nn.Dropout(p = 0.2),
                                 nn.Linear(400, len(class_to_idx)),
                                 nn.LogSoftmax(dim = 1))
    
    return model

def train(model, trainloader, validloader, criterion, learn_rate, epochs, device):
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    model.to(device)
    
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    
    with active_session():
        for epoch in range(epochs):
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    valid_loss = 0
                    accuracy = 0
                    with torch.no_grad():
                        for images, labels in validloader:
                            images, labels = images.to(device), labels.to(device)
                            logps = model(images)
                            batch_loss = criterion(logps, labels)
                            valid_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_ps, top_class = ps.topk(1, dim = 1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor))
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(validloader):.3f}")

                    running_loss = 0
                    model.train()
    
    return model, optimizer

def test_model(model, testloader, device, criterion):
    model.eval()
    model.to(device)
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim = 1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor))

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")
    
def input_args():
    parser = argparse.ArgumentParser(description = 'Getting inputs for training NN')
    parser.add_argument('data_directory', help = 'Path to directory containing datasets')
    parser.add_argument('--save_dir', help = 'Directory to save the checkpoints of model')
    parser.add_argument('--arch', help = 'Network Architecture to use options(vgg16, densenet121)')
    parser.add_argument('--learning_rate', help = 'Learning rate to use')
    parser.add_argument('--hidden_units', help = 'No of hidden units')
    parser.add_argument('--epochs', help = 'No of epochs to train')
    parser.add_argument('--gpu', help = 'Whether to use gpu for training or not', action = 'store_true')
    
    return parser.parse_args()

def main():
    in_args = input_args()
    
    save_dir = '' if in_args.save_dir is None else in_args.save_dir
    arch = 'vgg16' if in_args.arch is None else in_args.arch
    learning_rate = 0.001 if in_args.learning_rate is None else float(in_args.learning_rate)
    hidden_units = 25088 // 2 if in_args.hidden_units is None else int(in_args.hidden_units)
    epochs = 2 if in_args.epochs is None else int(in_args.epochs)
    gpu = False if in_args.gpu is None else True
    
    trainloader, validloader, testloader, class_to_idx = data_loaders(in_args.data_directory)
    
    model = build_model(arch, hidden_units, class_to_idx)
    
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    criterion = nn.NLLLoss()
    
    model, optimizer = train(model, trainloader, validloader, criterion, learning_rate, epochs, device)
    test_model(model, testloader, device, criterion)
    model.class_to_idx = class_to_idx
    
    checkpoint = {'epochs': epochs,
              'pretrained_model': arch,
              'batch_size': 64,
              'learn_rate': learning_rate,
              'hidden_units': hidden_units,
              'optimizer_state': optimizer.state_dict(),
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': class_to_idx}
    
    torch.save(checkpoint, './' + save_dir + '/checkpoint.pth')
    
    

if __name__ == '__main__':
    main()
    