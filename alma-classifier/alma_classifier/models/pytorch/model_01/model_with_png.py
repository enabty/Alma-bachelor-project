import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import timeit
from PIL import Image
import random
from scipy.ndimage import rotate
from skimage.transform import resize


n_epochs = 5

# Define the CNN model

################################### CLASS ###################################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(8192, 128)  # Are these optimal?
        self.fc2 = nn.Linear(128, 2)  # Are these optimal?

#  TODO: Add more layers when dataset is larger
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#****************************************************************************


################################### TRF ###################################
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.406], std=[0.225])
])


def linear_transformation(fits):
    ret_image = fits

    # random_resize = random.randint(150, 350)
    # ret_image = resize(ret_image, (random_resize, random_resize))

    ret_image = rotate(ret_image, random.randint(0, 360), reshape=False)

    if random.getrandbits(1):
        ret_image = np.fliplr(ret_image)
    if random.getrandbits(1):
        ret_image = np.flipud(ret_image)

    (lwr_bound, upr_bound) = int(random_resize/2) - 55, int(random_resize/2) - 45
    x = random.randint(lwr_bound, upr_bound)
    y = random.randint(lwr_bound, upr_bound)

    return ret_image[x:x+100, y:y+100]

# Load the positive and negative data from the .npy files
pos_data = np.load('./data/fits/pos_dataset.npy')
neg_data = np.load('./data/fits/pos_dataset.npy')


pos_data = np.array([linear_transformation(fits)for fits in pos_data if fits.shape == (400, 400)])


neg_data = ([linear_transformation(fits)for fits in neg_data if fits.shape == (400, 400)])

# y = [0] * len(fits_neg) + [1] * len(fits_pos)
# X = np.concatenate((fits_neg, fits_pos), axis=0)

# return X, y



























# # Normalize the data using the defined transforms
# pos_normalized = []
# for img in pos_data:
#     img_pil = Image.fromarray(img)
#     pos_normalized.append(transform(img_pil))
# pos_tensor = torch.stack(pos_normalized)
# pos_labels = torch.ones(pos_tensor.shape)

# neg_normalized = []
# for img in neg_data:
#     img_pil = Image.fromarray(img)
#     neg_normalized.append(transform(img_pil))
# neg_tensor = torch.stack(neg_normalized)
# neg_labels = torch.zeros(neg_tensor.shape)

# # Concatenate the positive and negative tensors and labels
# data = torch.cat((pos_tensor, neg_tensor), dim=0)
# labels = torch.cat((pos_labels, neg_labels), dim=0)

# # Create a TensorDataset from the data and labels
# dataset = TensorDataset(data, labels)

# # Create a DataLoader from the dataset for training the CNN
# train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

#****************************************************************************


################################### LOAD ###################################
# # Load the dataset
# train_set = datasets.ImageFolder(
#     root='./data/png/train/', transform=transform)
# num_classes = len(train_set.classes)
# # print(num_classes)  # 2
# train_loader = torch.utils.data.DataLoader(
#     train_set, batch_size=2, shuffle=True)
#****************************************************************************


################################### CREATE ###################################
# Create the model
model = CNN()

# Define the loss function and optimizer
# Maybe do weighted entropy loss function towards the smaller class in train data
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9)  # lr=0.001, 0.01, 0.1
#****************************************************************************


################################### TRAIN ###################################
t_0 = timeit.default_timer()  # Timer start
print("Training started")
# Train the model
# n_epochs = 10
total_images = 0
print(f'Number of epochs: {n_epochs}')
print('*'*50)
for epoch in range(n_epochs):  # Trying between 1-15 bc hardware bottleneck
    running_loss = 0.0
    epoch_images = 0  # count images trained in this epoch
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_images += len(inputs)  # increment counter by batch size
        total_images += len(inputs)  # increment total counter
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Epoch %d trained on %d images' % (epoch + 1, epoch_images))

print('*'*50)
print('Finished Training')
t_1 = timeit.default_timer()
# calculate elapsed time and print
elapsed_time = round((t_1 - t_0), 3)
print(f"Elapsed time: {elapsed_time} s")
#****************************************************************************


################################### TEST ###################################
test_set = datasets.ImageFolder(
    root='./data/png/test/', transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=10, shuffle=False)

# Init some useful variables
correct = 0
total = 0
predictions = []
true_labels = []
misclassified = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.numpy())
        true_labels.extend(labels.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # check for misclassified images and add their fpaths to the misclassified list
        for i in range(len(predicted)):
            if predicted[i] != labels[i]:
                filename = test_set.samples[i][0]
                true_label = test_set.classes[labels[i]]
                predicted_label = test_set.classes[predicted[i]]
                misclassified.append((filename, true_label, predicted_label))

print('*'*50)
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
print('*'*50)
#****************************************************************************


################################### PRINT MC ###################################
# print out the misclassified images and their true/predicted labels
print("Misclassified images:")
for item in misclassified:
    print('-'*35)
    print('Image:', item[0])
    print('True label:', item[1])
    print('Predicted label:', item[2])

print('*'*50)
print("Confusion Matrix (CLI):")
# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print(conf_matrix)
class_names = train_set.classes
print(class_names)

# Create the confusion matrix object
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, display_labels=class_names)

# Generate the plot of the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
cm_display.plot(ax=ax)
plt.show()
#****************************************************************************
