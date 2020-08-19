import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pylab import *
from PIL import Image
import sys

def detect(image_name, level=[200]):
    fig, ax = plt.subplots(1, 1)
    # read image to array, then get image border with contour
    im = array(Image.open(image_name).convert('L'))
    contour_plot = ax.contour(im, levels=level, colors='black', origin='image')
    plt.close()
    # get contour path
    contour_path = contour_plot.collections[0].get_paths()[0] # It has to be a polygon.
    x_table, y_table = contour_path.vertices[:, 0], contour_path.vertices[:, 1]
    # center the image
    x_table = x_table - min(x_table)
    y_table = y_table - min(y_table)
    x_table = x_table - max(x_table) / 2
    y_table = y_table - max(y_table) / 2
    return x_table, y_table

# Discrete Fourier Transform Series
def DTFS(z, k):
    N = len(z)
    ak = 0+0j
    for n in range(N):
        ak += z[n] * (np.cos(-2*np.pi/N*n*k) + 1j*np.sin(-2*np.pi/N*n*k))
    ak = ak/N
    return ak

if __name__ == "__main__":
    filename = None
    m = 25
    save = False
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-f':
            filename = sys.argv[i+1]
        elif sys.argv[i] == '-n':
            m = int(sys.argv[i+1])
        elif sys.argv[i] == '-s':
            save = True
    if filename == None:
        print("Please enter the filename. E.g. xxx.png ")
        sys.exit()
    else:
        fig, ax = plt.subplots(1,1)
        plt.axis('equal')
        x_table, y_table = detect(filename)
        y_table = y_table.tolist()
        x_table = x_table.tolist()
        count = 0
        sparse_y = []
        sparse_x = []
        for t in y_table:
            if count % 5 == 0 : # we do not need too much points
                sparse_y.append(t)
            count += 1    
        count = 0
        for t in x_table:
            if count % 5 == 0 : # we do not need too much points
                sparse_x.append(t)
            count += 1
        x = np.array(sparse_x)
        y = np.array(sparse_y)
        z = x + 1j * y
        cn = []
        for k in range(-m, m+1):
            cn.append(DTFS(z,k))
        T = np.linspace(0, 2*np.pi, 500)
        px = []
        py = []
        for t in T:
            p = 0+0j
            for j in range(2*m+1):
                p += cn[j] * ( np.cos((j-m)*t) + 1j*np.sin((j-m)*t) )
            px.append(p.real)
            py.append(p.imag)
        ax.plot(px, py, linewidth=2, color='blue')
        ax.plot(sparse_x,sparse_y, linewidth=2, color='black')
        n = np.linspace(-m,m,2*m+1)
        cn = np.array(cn)
        r = abs(cn)
        p = np.angle(cn)
        w = 2 * np.pi * n / len(z)
        circles = []
        dots = []
        for i in range(len(n)):
            circle, = ax.plot([], [], linewidth=1, color='grey')
            if i == len(n)-1:
                dot, = ax.plot([], [], 'o', color='red')
            else:
                dot, = ax.plot([], [], 'o', color='grey')
            circles.append(circle)
            dots.append(dot)
        theta = np.linspace(0, 2*np.pi, 100)
        def Anim(t):
            center = [0,0]
            for i in range(len(n)):
                circles[i].set_data(center[0]+r[i]*np.cos(theta), center[1]+r[i]*np.sin(theta))
                center = [center[0] + r[i]*np.cos(p[i]+w[i]*t), center[1]+r[i]*np.sin(p[i]+w[i]*t)]
                dots[i].set_data([center[0],center[1]])
        anim = animation.FuncAnimation(fig, Anim, frames=len(z), interval=50)
    if save:
        print("Please wait...")
        anim.save('result.gif', writer='imagemagick') # you need to install imagemagick to save gif
        sys.exit()
    plt.show()