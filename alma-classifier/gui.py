import tkinter as tk
from tkinter import *
from tkinter import filedialog
import customtkinter as ctk
from pathlib import Path
import sys
from main import generate_pos_training_data, generate_neg_training_data, classify_data


class MyTabView(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # create tabs
        self.add("Create Training Data")
        self.add("Create Neural Network")
        self.add("Classify Data on Neural Network")


        # Widgets for tab "Create Training Data"

        #------------------------------------------- Create Training Data -------------------------------------------#
        self.label = ctk.CTkLabel(master=self.tab("Create Training Data"), text="Browse to a folder where you want to save the .NPY file", anchor='center')
        self.label.grid(row=0, column=0, padx=20, pady=10)

        self.label_npy_file = ctk.CTkLabel(master=self.tab("Create Training Data"),
                                            width=100,
                                            text="NPY file:",
                                            fg_color="black",
                                            corner_radius=5)
        self.label_npy_file.grid(row=1, column=0, padx=20, pady=10)

        self.label_npy_file = ctk.CTkButton(master=self.tab(
            "Create Training Data"), text="Browse", command=self.browse_npy_file)
        self.label_npy_file.grid(row=2, column=0, padx=20, pady=10)
            
        self.run = ctk.CTkButton(master=self.tab("Create Training Data"), text="Run", command=self.run_1)
        self.run.grid(row=3, column=0, padx=20, pady=10)

        #------------------------------------------- Create Neural Network -------------------------------------------#

        self.label = ctk.CTkLabel(master=self.tab("Create Neural Network"), text="Here you can choose from idfferent parameters for your neural network and train it as well as save it for later use.", anchor='center')
        self.label.grid(row=0, column=0, padx=20, pady=10)

        #OPTIMIZER
        self.optimizer = ctk.StringVar(master=self.tab("Create Neural Network"),  value="adam")

        self.optimizer_label = ctk.CTkLabel(master=self.tab("Create Neural Network"), text="Optimizer:")
        self.optimizer_label.grid(row=1, column=0, padx=20, pady=10)

        self.optimizer = ctk.CTkOptionMenu(master=self.tab("Create Neural Network"), values=["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"], variable=self.optimizer)
        self.optimizer.grid(row=1, column=1, padx=20, pady=10)
        
        # #LOSS FUNCTION
        self.loss_function = ctk.StringVar(master=self.tab("Create Neural Network"), value="binary_crossentropy")

        self.loss_function_label = ctk.CTkLabel(master=self.tab("Create Neural Network"), text="Loss Function:")
        self.loss_function_label.grid(row=2, column=0, padx=20, pady=10)

        self.loss_function = ctk.CTkOptionMenu(master=self.tab("Create Neural Network"), values=["binary_crossentropy", "categorical_crossentropy", "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh", "huber_loss", "categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"], variable=self.loss_function)
        self.loss_function.grid(row=2, column=1, padx=20, pady=10)

        self.metrics = ctk.StringVar(master=self.tab("Create Neural Network"), value="accuracy")

        self.metrics_label = ctk.CTkLabel(master=self.tab("Create Neural Network"), text="Metrics:")
        self.metrics_label.grid(row=3, column=0, padx=20, pady=10)

        self.metrics = ctk.CTkOptionMenu(master=self.tab("Create Neural Network"), values=["accuracy", "binary_accuracy", "categorical_accuracy", "sparse_categorical_accuracy", "top_k_categorical_accuracy", "sparse_top_k_categorical_accuracy"], variable=self.metrics)
        self.metrics.grid(row=3, column=1, padx=20, pady=10)

        self.directory_labe     = ctk.CTkLabel(master=self.tab("Create Neural Network"), text="Choose directory to save the CNN:")
        self.directory_labe.grid(row=4, column=0, padx=20, pady=10)

        self.label_directory = ctk.CTkButton(master=self.tab("Create Neural Network"), text="Browse", command=self.browse_directory)
        self.label_directory.grid(row=4, column=1, padx=20, pady=10)


    # #------------------------------------------- Classify Data Func -------------------------------------------#

    def browse_npy_file(self):
        npy_file = filedialog.askopenfilename(initialdir=Path(sys.executable).parent,
                                        title="Select a .NPY file",
                                        filetypes=(("NPY files", "*.npy"), ("all files", "*.*")))
        self.label_npy_file.configure(text= "You choose following .NPY file:" + npy_file)

    def run_1(self):
        if self.label_npy_file.text == "NPY file:":
            print("Please select a .NPY file")
        else:
            generate_pos_training_data(self.label_npy_file.text)

    # ------------------------------------------- Create NN Func -------------------------------------------#

    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=Path(sys.executable).parent,
                                        title="Select a directory to store the CNN",
                                        mustexist=True)
        self.label_directory.configure(text= "You choose following directory:" + directory)

    def run_2(self):
        if self.label_directory.text == "Browse":
            print("Please select a directory")
        else:
            generate_pos_training_data(self.label_directory.text)



class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # create tabview
        self.tabview = MyTabView(master = self)
        self.tabview.pack(fill="both", expand=True)
        

        self.geometry("1000x800")

 

app = App()
app.mainloop()
