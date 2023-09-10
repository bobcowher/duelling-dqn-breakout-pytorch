import matplotlib.pyplot as plt
import os
from datetime import datetime


class LivePlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch x 10")
        self.ax.set_ylabel("Returns")
        self.ax.set_title("Returns over Epochs")

        self.data = None

        self.epochs = 0

    def update_plot(self, stats):
        self.data = stats['AvgReturns']
        self.eps_data = stats['EpsilonCheckpoint']

        self.epochs = len(self.data)

        # Update the plot limits and data
        self.ax.clear()  # Clear previous data
        self.ax.set_xlim(0, self.epochs)
        # self.ax.set_ylim(min(self.data) - 0.5, max(self.data) + 0.5)

        # Plot new data
        self.ax.plot(self.data, 'b-', label='Returns')
        self.ax.plot(self.eps_data, 'r-', label='Epsilon')
        self.ax.legend(loc="upper left")

        # Ensure the 'plots' directory exists
        if not os.path.exists('plots'):
            os.makedirs('plots')

        # Get the current date and time
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Save the plot to the plots directory with date appended
        self.fig.savefig(f'plots/plot_{current_date}.png')
