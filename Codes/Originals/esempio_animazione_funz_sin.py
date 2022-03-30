# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:55:20 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

#Creo la finestra, assi e oggetto da riempire successivamente.
fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)

#Creo la funzione init che "crea" l'animazione ed inizializza i dati.
def init():
    line.set_data([], [])
    return line,

#Creo l'animazione in funzione del numero di frame(i)
def animate(i):
    x = np.linspace(0, 4, 1000)
    y = np.sin(2*np.pi*x*i)+np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)


#anim.save('sine_wave.gif', writer='imagemagick')