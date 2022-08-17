
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import pycxsimulator

width = 400
height = width
T = 2.27


def initialize():
    global time, config, agents, empty, magnetization

    agents = []
    empty = []
    magnetization = []

    time = 0

    config = np.zeros([height, width])
    for x in range(width):
        for y in range(height):
            if random.random() < 0.5:
                config[x, y] = 1
            else:
                config[x, y] = -1


def observe():
    magnetization.append(sum(config))
    plt.cla()
    plt.imshow(config, vmin=-1, vmax=4, cmap=matplotlib.cm.hot)  # cm.bwr

    plt.axis('image')
    plt.title('t = ' + str(time))


def update():
    global time, config, agents, empty

    time += 1
    for count in range(int(width * height / 10)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        current_energy = (-1) * config[x, y] * (config[(x + 1) % width, y] + config[(
            x - 1) % width, y] + config[x, (y + 1) % height] + config[x, (y - 1) % height])
        if current_energy > 0:
            config[x, y] *= -1
        else:
            if random.random() < np.exp(2 * current_energy / T):
                config[x, y] *= -1


pycxsimulator.GUI().start(func=[initialize, observe, update])
