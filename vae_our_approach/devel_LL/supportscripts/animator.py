# First import everthing you need
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

class GifMaker:
    def __init__(self, fig, ax, path):
        self.fig = fig
        self.ax = ax

        print('Generating visualization of 3d latent space ....')
        # Animate
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                       frames=360, interval=20, blit=True)
        print('Saving generated gif into ', path)
        writergif = animation.PillowWriter(fps=30)
        anim.save(path, writer=writergif)

    def init(self):
        #self.ax.scatter([2,3], [4,5], [4,3], marker='o', s=20, c="goldenrod", alpha=0.6)
        return self.fig,

    def animate(self, i):
        self.ax.view_init(elev=10., azim=i)
        return self.fig,


fig = plt.figure()
ax = Axes3D(fig)
GifMaker(fig, ax, 'uloz.gif')