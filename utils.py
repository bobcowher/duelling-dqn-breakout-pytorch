# This file should be used to create troubleshooting and reporting utilities that aren't critical to fucntionality

import matplotlib.pyplot as plt

def display_observation_image(observation):
    observation = observation.squeeze(0)
    observation = observation.squeeze(0)
    plt.imshow(observation, cmap='gray')  # 'cmap' is color map; 'gray' for grayscale images
    plt.axis('off')  # To hide axis numbers
    plt.show()