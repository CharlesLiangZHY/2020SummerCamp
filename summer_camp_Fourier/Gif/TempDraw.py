import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import matplotlib.animation as animation

def FS(ft,k):
    dt = 1/len(ft)
    ak = 0+0j
    for i in range(len(ft)):
        ak+=ft[i]*(np.cos(-2*np.pi*k*dt*i) + 1j*np.sin(-2*np.pi*k*dt*i))*dt
    return ak

'''
Draw 4 subplots to show how the simulation gets closer to the original signal.
'''

T = 5
nums = [1, 3, 5, 15]

fig = figure(figsize = (6,8))

picts = [[],[],[],[]]
axes = [subplot2grid((4, 1), (0, 0)), subplot2grid((4, 1), (1, 0)), subplot2grid((4, 1), (2, 0)), subplot2grid((4, 1), (3, 0))]

for j in range(len(axes)):
    ax = axes[j]
    ax.set_xlim([-2, T])
    ax.set_ylim([-1, 1])
    m = nums[j]
    plt.axis('equal')


    cn = []
    n = np.linspace(-m,m,2*m+1)


    ax.set_xlim([-2, T])
    ax.set_ylim([-1, 1])


    '''
    Square Wave
    '''
    # sqwx = []
    # sqwy = []
    # for i in range(T):
    #     sqwx+=[i+0.5*d/50 for d in range(50)]
    #     sqwy+=[0.5 for d in range(50)]
    #     sqwx+=[i+0.5+0.5*d/50 for d in range(50)]
    #     sqwy+=[-0.5 for d in range(50)]

    # squarewave, = ax.plot(sqwx,sqwy, linewidth=1, color='black') 

    # for k in n:
    #     cn.append(FS(sqwy[:100], k))


    '''
    Oblique Wave
    '''
    # olwx = []
    # olwy = []
    # for i in range(T):
    #     olwx+=[i+0.5*d/50 for d in range(50)]
    #     olwy+=[0.5*d/50 for d in range(50)]
    #     olwx+=[i+0.5+0.5*d/50 for d in range(50)]
    #     olwy+=[-1+0.5+0.5*d/50 for d in range(50)]

    # obliquewave, = ax.plot(olwx,olwy, linewidth=1, color='black') 

    # for k in n:
    #     cn.append(FS(olwy[:100], k))

    '''
    Sharp Wave
    '''
    # spwx = []
    # spwy = []
    # for i in range(T):
    #     spwx+=[i+0.5*d/50 for d in range(50)]
    #     spwy+=[0.5*d/50 for d in range(50)]
    #     spwx+=[i+0.5+0.5*d/50 for d in range(50)]
    #     spwy+=[1-0.5-0.5*d/50 for d in range(50)]

    # shapewave, = ax.plot(spwx,spwy, linewidth=1, color='black') 

    # for k in n:
    #     cn.append(FS(spwy[:100], k))


    '''
    Strange Wave
    '''
    stwx = []
    stwy = []
    for i in range(T):
        stwx+=[i+0.25*d/50 for d in range(50)]
        stwy+=[0.25*d/50 for d in range(50)]

        stwx+=[i+0.25+0.25*d/50 for d in range(50)]
        stwy+=[0.25 for d in range(50)]

        stwx+=[i+0.5+0.25*d/50 for d in range(50)]
        stwy+=[1-0.75*d/50 for d in range(50)]

        stwx+=[i+0.75+0.25*d/50 for d in range(50)]
        stwy+=[-0.5+0.25*d/50 for d in range(50)]

    strangewave, = ax.plot(stwx,stwy, linewidth=1, color='black') 

    for k in n:
        cn.append(FS(stwy[:200], k))



    cn = np.array(cn)
    r = abs(cn)
    p = np.angle(cn)
    w = 2 * np.pi * n

    circles = []
    dots = []
    sins = []
    sinp =[]
    for i in range(len(n)):
        circle, = ax.plot([], [], linewidth=1, color='grey')
        if i == len(n)-1:
            dot, = ax.plot([], [], 'o', color='red')
        else:
            dot, = ax.plot([], [], 'o', color='grey')
        sin, = ax.plot([], [], linewidth=1, color='grey', linestyle='--')
        circles.append(circle)
        dots.append(dot)
        sins.append(sin)
        sinp.append([[],[]])


    wave, = ax.plot([], [], linewidth=1, color='blue') 
    line, = ax.plot([], [], linewidth=1, color='blue', linestyle='--')

    px = []
    py = []

    picts[j].append(r) # 0
    picts[j].append(p) # 1
    picts[j].append(w) # 2
    picts[j].append(circles) # 3
    picts[j].append(dots) # 4
    picts[j].append(sins) # 5
    picts[j].append(sinp) # 6
    picts[j].append(wave) # 7
    picts[j].append(line) # 8
    picts[j].append(px) # 9
    picts[j].append(py) # 10
    picts[j].append(n) # 11

def Anim(t):
    dt = t/100
    theta = np.linspace(0, 2*np.pi, 100)
    for j in range(4):
        r = picts[j][0]
        p = picts[j][1]
        w = picts[j][2]
        circles = picts[j][3]
        dots = picts[j][4]
        sins = picts[j][5]
        sinp = picts[j][6]
        wave = picts[j][7]
        line = picts[j][8]
        px = picts[j][9]
        py = picts[j][10]
        n = picts[j][11]



        center = [0,0]
        for i in range(len(n)):
            circles[i].set_data( center[1]+r[i]*np.sin(theta), center[0]+r[i]*np.cos(theta))
            center = [center[0]+r[i]*np.cos(p[i]+w[i]*dt), center[1] + r[i]*np.sin(p[i]+w[i]*dt)]
            dots[i].set_data([center[1],center[0]])
            sinp[i][0].append(dt)
            sinp[i][1].append(r[i]*np.cos(p[i]+w[i]*dt))
            sins[i].set_data(sinp[i][0], sinp[i][1])
        px.append(dt)
        py.append(center[0])
        wave.set_data(px, py)
        line.set_data([center[1],dt],[center[0],center[0]])

anim = animation.FuncAnimation(fig, Anim, frames=T*100, interval=25, repeat=False)

anim.save('Result.gif' , writer='imagemagick')

# plt.show()