'''
Assignment 2
Student: STEFANO ANDREOTTI
'''

# *** Packages ***
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
from math import floor
import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    # Function to print an image
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def training(model, optimizer, loss_fn, DEVICE: torch.device, n_epochs: int, batch_size: int, trainloader: DataLoader,
             validloader: DataLoader, verbose: bool=False):
    # Function to train a model, uses verbose for more detailed messages on train
    model = model.to(DEVICE)

    train_count = 0
    train_loss_list = []
    validation_loss_list = []

    for epoch in range(n_epochs):
        loss_train = 0
        correct_train = 0

        for data, target in trainloader:
            # Set the model in training mode
            model.train()
            data, target = data.to(DEVICE), target.to(DEVICE)
            # Set the gradient to 0
            optimizer.zero_grad()
            # Make a prediction
            output = model(data)
            # Compute the loss function
            loss = loss_fn(output, target)
            loss_train += loss.item()
            # Backpropagation
            loss.backward()
            # Update parameters
            optimizer.step()
        
            # Calculate batch loss and batch accuracy
            _, predicted = torch.max(output.data, 1)  # Get the index of the maximum value
            correct_train += (predicted == target).sum().item()  # Count correct predictions
            train_count += len(data)  # Update train_count for accurate batch calculation

            # Print train loss and accuracy every n batches
            if verbose:
                if (train_count // batch_size) % ((len(trainloader)-1) // 3) == 0:
                    train_accuracy = 100.0 * correct_train / train_count
                    print(f"Epoch: {epoch + 1}, Batch {train_count // batch_size}: "
                            f"Train Loss: {loss_train / train_count:.4f}, Train Accuracy: {train_accuracy:.2f} %")

        train_accuracy = 100.0 * correct_train / train_count
        loss_train = loss_train / len(trainloader)
        train_loss_list.append(loss_train)
        print(f"Epoch: {epoch + 1}, Train Loss: {loss_train:.4f}, Train Accuracy: {train_accuracy:.2f} %")

        # Reset counters for next epoch
        train_count = 0  # Reset train_count for accurate batch calculation in next epoch
        
        # Validation at the end of each epoch
        with torch.no_grad():
            model.eval()
            loss_valid = 0
            correct_valid = 0

            for data, target in validloader:
                data, target = data.to(DEVICE), target.to(DEVICE)

                # Make a prediction
                output = model(data)

                # Compute the loss function
                validation_loss = loss_fn(output, target)
                loss_valid += validation_loss.item()

                # Calculate validation accuracy
                _, predicted = torch.max(output.data, 1)
                correct_valid += (predicted == target).sum().item()
                valid_total = target.size(0)

                # No batch print because valid has only 1 batch

            # Epoch accuracy and loss calculations
            validation_accuracy = 100 * (correct_valid / valid_total)
            loss_valid = loss_valid / len(validloader)
            validation_loss_list.append(loss_valid)
            print(f"Epoch: {epoch + 1}, Validation Loss: {loss_valid:.4f}, Validation Accuracy: {validation_accuracy:.2f} %\n")

    return model, train_loss_list, validation_loss_list


def accuracy(model, testloader: DataLoader):
    # Function for calculate accuracy on testset
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += target.size(0)
            n_correct += (predicted == target).sum().item()

        acc = 100.0 * n_correct / n_samples
    print("Accuracy on the test set:", acc, "%")


def loss_plot(n_epochs: int, train_loss_list: list, validation_loss_list: list, title: str):
    # Function that ploot 2 functions, train loss and validation loss
    plt.figure()
    plt.plot(range(1, n_epochs+1), train_loss_list)
    plt.plot(range(1, n_epochs+1), validation_loss_list)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.title(title)
    plt.savefig(title+".png", dpi=300, format="png")
    plt.show()


# *** Models ***
def out_dimensions(conv_layer, h_in: int, w_in: int):
    # Function to compute output dimension of convolutional layers
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out
    

'''
Q6 - Code
'''
class CNNS(nn.Module):
    # Definition of CNN
    def __init__(self):
        super(CNNS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv1, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv3, h_out, w_out)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv4, h_out, w_out)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        self.fc1 = nn.Linear(64 * h_out * w_out, 32)
        self.dimensions_final = (64, h_out, w_out)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # First Block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second Block
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Full connected
        n_channels, h, w = self.dimensions_final
        x = x.view(-1, n_channels * h * w)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


'''
Q9 -  Code
'''
class SSJCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(SSJCNN, self).__init__()
        # Input layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv1, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)

        # Second block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv3, h_out, w_out)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv4, h_out, w_out)
        self.bn4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)

        # Third block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv5, h_out, w_out)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv6, h_out, w_out)
        self.bn6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(128 * h_out * w_out, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.maxpool1(x)

        # Second block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.maxpool2(x)

        # Third block
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.gelu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.gelu(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn7(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = self.bn8(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)

    '''
    Q4 - Code
    '''
    # Initialize trasformer compose to cast images into tensors and normalize them
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0, std=1)])

    '''
    Q2 - Code
    '''
    # Creating trainloader in batches and testloader in one batch
    batch_size = 32
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformer)
    testloader = DataLoader(testset, batch_size=len(testset))
    print()

    # Get the labels and images
    labels = trainset.targets
    images = trainset.data

    # Find an image for each class
    class_images = [None] * 10
    for i in range(len(labels)):
        label = labels[i]
        if class_images[label] is None:
            class_images[label] = images[i]

    # Class labels
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Display the images
    fig, axs = plt.subplots(1, 10, figsize=(15, 5))
    for i in range(10):
        axs[i].imshow(class_images[i])  # Permute dimensions for correct display
        axs[i].set_title(f"Class: {classes[i]}")
        axs[i].axis('off')
    plt.show()

    # Count class distribution for training
    labels = trainset.targets
    train_counts = [0] * 10
    for label in labels:
        train_counts[label] += 1

    # Count class distribution for test
    labels = testset.targets
    test_counts = [0] * 10
    for label in labels:
        test_counts[label] += 1

    # Create the histogram for class distribution
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(classes))

    # Plot the class bars
    ax.bar(index, train_counts, bar_width, label='Train')
    ax.bar(index + bar_width, test_counts, bar_width, label='Test')

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.show()


    '''
    Q3
    '''
    # Print the type and size of images
    # Already tensors because i used transformer in loading the data
    first_image = images[0]

    print(f"Shape of the image tensor: {first_image.shape}")
    width, height, channels = first_image.shape
    print(f"Width: {width}, Height: {height}, Channels: {channels}\n")

    '''
    Q5 - Code
    '''
    # Split testset into test and validation
    validset, testset = torch.utils.data.random_split(testset, [0.5, 0.5])
    testloader = DataLoader(testset, batch_size=len(testset))
    validloader = DataLoader(validset, batch_size=len(validset))

    '''
    Q7 -  Code
    '''
    # Creating the first CNN
    model = CNNS()
    learning_rate = 0.033
    n_epochs = 4
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print("CNNS Training...")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model, train_loss_list, validation_loss_list = training(model= model, optimizer=optimizer, loss_fn=loss_fn,
                                                            DEVICE=DEVICE, n_epochs=n_epochs, batch_size=batch_size,
                                                            trainloader=trainloader, validloader=validloader,
                                                            verbose=True)
    # Print accuracy
    accuracy(model=model, testloader=testloader)
    print("\n")


    '''
    Q8 -  Code
    '''
    # Plotting the loss function for training and validation
    loss_plot(n_epochs=n_epochs, train_loss_list=train_loss_list, validation_loss_list=validation_loss_list,
              title='Train and validation loss')


    '''
    Q9 -  Code
    '''
    # Data augmentation
    data_augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])

    # Apply data augmentation
    trainset_q9 = datasets.CIFAR10(root='./data', train=True, download=False, transform=data_augment)
    trainloader_q9 = DataLoader(trainset_q9, batch_size=batch_size, shuffle=True)

    # Training the best CNN
    model = SSJCNN()
    learning_rate = 0.0283
    n_epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print("SSJCNN Training...")
    model, train_loss_list, validation_loss_list = training(model=model, optimizer=optimizer, loss_fn=loss_fn,
                                                            DEVICE=DEVICE, n_epochs=n_epochs, batch_size=batch_size,
                                                            trainloader=trainloader_q9, validloader=validloader,
                                                            verbose=False)
    # Print stats
    accuracy(model=model, testloader=testloader)
    loss_plot(n_epochs=n_epochs, train_loss_list=train_loss_list, validation_loss_list=validation_loss_list,
              title='Train and validation loss')

    '''
    Q10 -  Code
    '''
    learning_rate = 0.03
    n_epochs = 4

    for seed in range(5,10):
        torch.manual_seed(seed)
        print("Seed equal to ", torch.random.initial_seed())

        # Define model and train
        model = CNNS()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        model, train_loss_list, validation_loss_list = training(model= model, optimizer=optimizer, loss_fn=loss_fn,
                                                                DEVICE=DEVICE, n_epochs=n_epochs, batch_size=batch_size,
                                                                trainloader=trainloader, validloader=validloader,
                                                                verbose=False)
        accuracy(model=model, testloader=testloader)
        print("\n")