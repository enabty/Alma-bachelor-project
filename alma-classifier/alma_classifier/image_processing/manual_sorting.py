import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


"""

Class used to manually sort and view data.

"""
class DataViewer:
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
        self.index += 1
        if self.index >= len(self.matrices):
            print("No more matrices to show")
            self.root.quit()
        else:
            self.show_matrix()


""""

Sort_manually takes a numpy array of matrices displays them and gives the user the option to delete or save them.

Returns a numpy array of the saved  matrices.

"""


def sort_manually(data):
    return DataViewer(data).matrices
    
def save_data_to_npy(data, npy_file):
    np.save(npy_file, data, allow_pickle=True)

def load_data_from_npy(npy_file):
    return np.load(npy_file, allow_pickle=True)


__name__ == '__main__' and print('manual_sorting.py is working')
