import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import glob
import ntpath
import astropy.io.fits as fits
from .support_functions import *

"""

Class used to manually sort and view data.

"""


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class ManualSorterFITS:
    def __init__(self, fits_name_data):
        self.fits_name_data = fits_name_data
        self.data, self.name = zip(*fits_name_data)
        self.index = 0

        self.root = tk.Tk()
        self.root.geometry("500x600")
        self.root.title("Matrix Viewer")

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.imshow(self.data[self.index], cmap="CMRmap_r")
        self.ax.set_title('Following object was identified as positive  \n' + path_leaf(self.name[self.index]))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.delete_button = tk.Button(
            self.root, text="Delete", command=self.delete_matrix)
        self.delete_button.pack(side="right")

        self.save_button = tk.Button(
            self.root, text="Save", command=self.save_matrix)
        self.save_button.pack(side="right")

        self.root.mainloop()

    def show_data(self):
        self.ax.clear()
        self.ax.imshow(self.data[self.index], cmap="CMRmap_r")
        self.ax.set_title(
            'Following object was identified as positive  \n' + path_leaf(self.name[self.index]))
        self.canvas.draw()

    def delete_matrix(self):
        self.ax.clear()
        if self.index >= len(self.data):
            print("No more matrices to show")
            self.fits_files_data = np.array(list(zip(self.data, self.name)))
            self.root.quit()
        else:
            self.data = np.delete(self.data, self.index, axis=0)
            self.name = np.delete(self.name, self.index, axis=0)
            if self.index >= len(self.data):
                self.root.quit()
            else:
                self.show_data()

    def save_matrix(self):
        self.ax.clear()

        self.index += 1
        if self.index >= len(self.data):
            print("No more matrices to show")
            self.fits_files_data = np.array(list(zip(self.data, self.name)))
            self.root.quit()
        else:
            self.show_data()



""""

DataViewers and DataViewer can be used to view data in a GUI and sort. Viewers is used for
mnultiple matrices and Viewer is used for a single matrix. They do not however "know" what
they are looking at, they are mostly usefull in sorting trainingdata for a CNN. They could 
however be integreated in the class above so that they can be used for any data. But for now
ignore them :)


"""
class DataViewers:
    def __init__(self, matrices):
        self.matrices = matrices
        self.index = 0

        self.root = tk.Tk()
        self.root.geometry("500x600")
        self.root.title("Matrix Viewer")

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.imshow(self.matrices[self.index], cmap = "CMRmap_r" )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.delete_button = tk.Button(
            self.root, text="Delete", command=self.delete_matrix)
        self.delete_button.pack(side="right")

        self.save_button = tk.Button(
            self.root, text="Save", command=self.save_matrix)
        self.save_button.pack(side="right")

        self.root.mainloop()

    def show_matrix(self):
        self.ax.clear()
        self.ax.imshow(self.matrices[self.index], cmap = "CMRmap_r")
        self.canvas.draw()

    def delete_matrix(self):
        self.ax.clear()
        if self.index >= len(self.matrices):
            print("No more matrices to show")
            self.root.quit()
        else:
            self.matrices = np.delete(self.matrices, self.index, axis=0)
            if self.index >= len(self.matrices):
                self.root.quit()
            else:
                self.show_matrix()
        

    def save_matrix(self):
        self.ax.clear()

        self.index += 1
        if self.index >= len(self.matrices):
            print("No more matrices to show")
            self.root.quit()
        else:
            self.show_matrix()

class DataViewer:

    def __init__(self, matrix):
        self.matrix = matrix
        self.save = False

        self.root = tk.Tk()
        self.root.geometry("500x600")
        self.root.title("Matrix Viewer")

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.imshow(self.matrix, cmap="CMRmap_r")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.delete_button = tk.Button(
            self.root, text="Delete", command=self.delete_matrix)
        self.delete_button.pack(side="right")

        self.save_button = tk.Button(
            self.root, text="Save", command=self.save_matrix)
        self.save_button.pack(side="right")

        self.root.mainloop()

    def delete_matrix(self):
        self.save = False
        [plt.close(figures) for figures in plt.get_fignums()]
        print(plt.get_fignums())
        # plt.close(self.ax)
        self.root.quit()

    def save_matrix(self):
        self.save = True
        self.ax.clear()
        self.root.quit()


""""

Sort_manually takes a numpy array of matrices displays them and gives the user the option to delete or save them.

Returns a numpy array of the saved  matrices.

"""


def sort_multiple_manually(data):
    return DataViewers(data).matrices
    
# def sort_manually(data):
#     return DataViewer(data).save

def save_data_to_npy(data, npy_file):
    np.save(npy_file, data, allow_pickle=True)

def load_data_from_npy(npy_file):
    return np.load(npy_file, allow_pickle=True)


"""

Takes data and a NNTensorflowModel and predicts the class of each matrix in the data. The iamges are then 
classified by the network. If the network classified the image as positive it is displayed and the user can
choose to save it or delete it. If the network classified the image as negative it is deleted.

The function returns a numpy array of the saved matrices with the paths to the original fits files.
"""


def predict(model, fits): return model.predict(np.array([fits])).argmax(axis=-1)[0]


def predict_fits(file_path, model):
    fits_files_data = [(fits.getdata(file).squeeze(), file)for file in glob.glob(file_path + '/*.fits')]
    fits_files_data = [(crop_around_middle_50x50_percent(data), name) for (data, name) in fits_files_data if data.shape[0] > 500 and data.shape[1] > 500]
    fits_files_data = [(crop_around_max_value_400x400(data), name)for (data, name) in fits_files_data]
    fits_files_data = [((data[150:250, 150:250]), name) for (data, name) in fits_files_data if data.shape == (400, 400)]
    fits_files_data = [(data, name) for (data, name) in fits_files_data if predict(model, data) == 1]
    return ManualSorterFITS(fits_files_data).fits_files_data




__name__ == '__main__' and print('manual_sorting.py is working')
