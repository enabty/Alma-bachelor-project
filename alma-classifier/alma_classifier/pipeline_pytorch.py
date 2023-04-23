import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import timeit
# import numpy as np
# import matplotlib.image as mpimg
# import torch.optim as optim
## Own imports
from models.pytorch.model_01 import model_architecture, model_transf, constants



n_epochs = 5

def init():
# Create the model
    model = model_architecture.CNN()
