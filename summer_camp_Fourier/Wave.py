import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

def FS(ft,k):
    dt = 1/len(ft)
    ak = 0+0j
    for i in range(len(ft)):
        ak+=ft[i]*(np.cos(-2*np.pi*k*dt*i) + 1j*np.sin(-2*np.pi*k*dt*i))*dt
    return ak

if __name__ == "__main__":
    T = 5
    m = 3 
    WaveType = 4 # 1 Square Wave 2 Oblique Wave 3 Sharp Wave
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-t':
            if sys.argv[i+1] == 'square':
                WaveType = 1
            elif sys.argv[i+1] == 'oblique':
                WaveType = 2
            elif sys.argv[i+1] == 'sharp':
                WaveType = 3
        elif sys.argv[i] == '-n':
            m = int(sys.argv[i+1]) 
    cn = []
    n = np.linspace(-m,m,2*m+1)
    fig, ax = plt.subplots(1,1)
    plt.axis('equal')
    ax.set_xlim([-2, T])
    ax.set_ylim([-1, 1])
    '''
    Square Wave
    '''
    if WaveType == 1:
        sqwx = []
        sqwy = []
        for i in range(T):
            sqwx+=[i+0.5*d/50 for d in range(50)]
            sqwy+=[0.5 for d in range(50)]
            sqwx+=[i+0.5+0.5*d/50 for d in range(50)]
            sqwy+=[-0.5 for d in range(50)]
        squarewave, = ax.plot(sqwx,sqwy, linewidth=1, color='black') 
        for k in n:
            cn.append(FS(sqwy[:100], k))
    '''
    Oblique Wave
    '''
    if WaveType == 2:
        olwx = []
        olwy = []
        for i in range(T):
            olwx+=[i+0.5*d/50 for d in range(50)]
            olwy+=[0.5*d/50 for d in range(50)]
            olwx+=[i+0.5+0.5*d/50 for d in range(50)]
            olwy+=[-1+0.5+0.5*d/50 for d in range(50)]
        obliquewave, = ax.plot(olwx,olwy, linewidth=1, color='black') 
        for k in n:
            cn.append(FS(olwy[:100], k))
    '''
    Sharp Wave
    '''
    if WaveType == 3:
        spwx = []
        spwy = []
        for i in range(T):
            spwx+=[i+0.5*d/50 for d in range(50)]
            spwy+=[0.5*d/50 for d in range(50)]
            spwx+=[i+0.5+0.5*d/50 for d in range(50)]
            spwy+=[1-0.5-0.5*d/50 for d in range(50)]
        shapewave, = ax.plot(spwx,spwy, linewidth=1, color='black') 
        for k in n:
            cn.append(FS(spwy[:100], k))
    '''
    Strange Wave
    '''
    if WaveType == 4:
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
    theta = np.linspace(0, 2*np.pi, 100)
    px = []
    py = []
    def Anim(t):
        dt = t/100
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
    plt.show()