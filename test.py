# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# 22000 was a good epoch.

from breakout import *
from utils import *
from agent import *
from model import *
import torch
import os

import sys

# Set default value for input_file
input_file = 'latest.pt'

# Check if there are enough arguments to get the value for input_file
if len(sys.argv) > 1:
    input_file = sys.argv[1]

input_file = 'models/' + input_file

print(f'Using file: {input_file}')


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AtariNet(nb_actions=4)

model.load_the_model(weights_filename=input_file)

agent = Agent(model=model,
              device=device,
              epsilon=0.01,
              min_epsilon=0.01,
              nb_warmup=50, # originally 10000
              nb_actions=4,
              memory_capacity=20000,
              batch_size=32)

test_environment = DQNBreakout(device=device, render_mode='human')

agent.test(env=test_environment)



