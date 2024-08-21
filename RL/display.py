import matplotlib.pyplot as plt
import numpy as np

class DisplayWindow:
    #TODO: Finish this
    def __init__(self, rows=2, cols=2, figsize=(10, 8)):
        self.fig, self.axs = plt.subplots(rows, cols, figsize=figsize)
        
        # Initialize the lines to be updated later

        
    def update_plots(self, new_data_x, new_data_y):
        plt.pause(0.1)

    def show(self):
        # Display the final plot
        plt.show()