# =-=-=-=-=-=-=-=-=-=-=-=- module -=-=-=-=-=-=-=-=-=-=-=-= #
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math, os, argparse

sns.set_style('whitegrid')

# =-=-=-=-=-=-=-=-=-=-=-=- main -=-=-=-=-=-=-=-=-=-=-=-= #
def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default='result/anime3.gif')
    args = parser.parse_args()

    # make result directory
    if not os.path.exists('result/'):
        os.makedirs('result/')

    # define plot function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    def plot(frame):
        """update function for graph"""
        # clean graph
        plt.cla()
        # add data
        x.append(frame)
        y.append(math.sin(frame))
        z.append(math.cos(frame))
        # replot
        ax.scatter(x, y, z)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


    # visualize result
    anime = animation.FuncAnimation(
        fig, plot,
        interval=10, # [ms]
        frames=np.arange(0, 10, 0.1),
        repeat=False,
    )
    anime.save(args.result, writer='pillow')
    plt.pause(.1)

if __name__ == '__main__':
    main()