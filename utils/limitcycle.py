import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

#import warnings
#warnings.resetwarnings()
#warnings.simplefilter('error')


def func_circle(t, xy, dtheta=0.1, lam=0.1):
    x = xy[0]
    y = xy[1]
    dxdt = dtheta*(lam*x-y - lam*x*(x*x+y*y))
    dydt = dtheta*(x+lam*y - lam*y*(x*x+y*y))
    return dxdt, dydt


def circle(dtheta=0.1, lam=0.1):
    return lambda t, xyz: func_circle(t, xyz, dtheta=dtheta, lam=lam)


def StuartLandau(t, xy):
    x = xy[0]
    y = xy[1]
    dxdt = x-2*np.pi*y - (x-y)*(x*x+y*y)
    dydt = 2*np.pi*x+y - (x+y)*(x*x+y*y)
    return dxdt, dydt


def func_vanderPol(t, xy, mu=0.2):
    x = xy[0]
    y = xy[1]
    dxdt = y
    dydt = mu*(1-x*x) * y - x
    return dxdt, dydt


def vanderPol(mu=0.2):
    return lambda t, xyz: func_vanderPol(t, xyz, mu=mu)


def func_Rossler(t, xyz, c=4):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    a, b = 0.1, 0.1
    dxdt = -y-z
    dydt = x + a*y
    dzdt = b + x*z - c*z
    return dxdt, dydt, dzdt


def Rossler(c=4):
    return lambda t, xyz: func_Rossler(t, xyz, c=c)

def func_fhn(t,xy):
    x = xy[0]
    y = xy[1]
    a,b,c = 0.2, 0.5, 10.0
    dxdt = c*(x-x**3 -y)
    dydt = x-b*y+a
    return dxdt, dydt

def fhn():
    return lambda t,xyz : func_fhn(t, xyz)

def func_fhn3(t,xy):
    x = xy[0]
    y = xy[1]
    e,a,b,I = 0.08,0.7, 0.8,0.8
    dxdt = x-(x**3)/3-y+I
    dydt = e*(x+a-b*y)
    return dxdt, dydt

def fhn3():
    return lambda t,xyz : func_fhn3(t, xyz)

def func_fhn2(t,xys):
    x1 = xys[0]
    y1 = xys[1]
    x2 = xys[2]
    y2 = xys[3]
    e,a,b,I = 0.08,0.7,0.8,0.4
    dx1dt = e*(y1+a-b*x1)
    dy1dt = y1-y1**3/3-x1+I
    dx2dt = e*(y2+a-b*x2)
    dy2dt = y2-y2**3/3-x2+I
    return dx1dt, dy1dt, dx2dt, dy2dt

def alpha_m(V):
    return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))
    
def beta_m(V):
    return 4.0*np.exp(-(V+65.0) / 18.0)

def alpha_h(V):
    return 0.07*np.exp(-(V+65.0) / 20.0)

def beta_h(V):
    return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

def alpha_n(V):
    return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

def beta_n(V):
    return 0.125*np.exp(-(V+65) / 80.0)

def func_HodgkinHuxley(t, vec,
                       input_current=30.0,
                       C=1.0,
                       G_Na=120.0,
                       G_K=36.0,
                       G_L=0.3,
                       E_Na=50.0,
                       E_K=-77.0,
                       E_L=-54.4):
    v = vec[0]
    m = vec[1]
    h = vec[2]
    n = vec[3]
    dvdt = (G_Na*(m**3)*h*(E_Na-v)
            + G_K*(n**4)*(E_K-v)
            + G_L*(E_L-v)
            + input_current
            )/C
    dmdt = alpha_m(v) * (1.0 - m) - beta_m(v) * m
    dhdt = alpha_h(v) * (1.0 - h) - beta_h(v) * h
    dndt = alpha_n(v) * (1.0 - n) - beta_n(v) * n
    return dvdt, dmdt, dhdt, dndt


def HodgkinHuxley():
    return lambda t, vec: func_HodgkinHuxley(t, vec)


def fhn2():
    return lambda t, xys: func_fhn2(t, xys)


def make_StuartLandau_inital_state(n=10):
    xy = []
    for x in np.linspace(-2, 2, n):
        for y in np.linspace(-2, 2, n):
            xy.append([x, y])
    return xy


def func_WillamowskiRossler(t, xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    b1, b2 = 80, 20
    d1, d2, d3 = 0.16, 0.13, 16
    dxdt = x*(b1-d1*x-y-z)
    dydt = y*(b2-d2*y-x)
    dzdt = z*(x-d3)
    return dxdt, dydt, dzdt


def func_fhn_ring(t, xy):
    N = 10
    A = 0.7
    B = 0.8
    E = 0.08
    I = 0.32
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if (i-j) in [1, -9]:
                K[i][j] = 0.3
            if (i-j) in [-1, 9]:
                K[i][j] = -0.3
    duv = np.zeros(2*N)
    for i in range(N):
        duv[i] = E*(xy[i+N]+A-B*xy[i])
        duv[i+N] = xy[i+N] - xy[i+N]**3/3 - xy[i] + I
    for i in range(N):
        for j in range(N):
            duv[i+N] += K[i][j]*(xy[j+N]-xy[i+N])

    return duv


def func_fhn_ring6(t, xy):
    N = 6
    A = 0.7
    B = 0.8
    E = 0.08
    I = 0.32
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if (i-j) in [1, -N+1]:
                K[i][j] = 0.3
            if (i-j) in [-1, N-1]:
                K[i][j] = -0.3
    duv = np.zeros(2*N)
    for i in range(N):
        duv[i] = E*(xy[i+N]+A-B*xy[i])
        duv[i+N] = xy[i+N] - xy[i+N]**3/3 - xy[i] + I
    for i in range(N):
        for j in range(N):
            duv[i+N] += K[i][j]*(xy[j+N]-xy[i+N])

    return duv

def func_fhn_ring3(t, xy):
    N = 3
    A = 0.7
    B = 0.8
    E = 0.08
    I = 0.32
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if (i-j) in [1, -N+1]:
                K[i][j] = 0.3
            if (i-j) in [-1, N-1]:
                K[i][j] = -0.3
    duv = np.zeros(2*N)
    for i in range(N):
        duv[i] = E*(xy[i+N]+A-B*xy[i])
        duv[i+N] = xy[i+N] - xy[i+N]**3/3 - xy[i] + I
    for i in range(N):
        for j in range(N):
            duv[i+N] += K[i][j]*(xy[j+N]-xy[i+N])

    return duv

def func_fhn_ring4(t, xy):
    N = 4
    A = 0.7
    B = 0.8
    E = 0.08
    I = 0.32
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if (i-j) in [1, -N+1]:
                K[i][j] = 0.3
            if (i-j) in [-1, N-1]:
                K[i][j] = -0.3
    duv = np.zeros(2*N)
    for i in range(N):
        duv[i] = E*(xy[i+N]+A-B*xy[i])
        duv[i+N] = xy[i+N] - xy[i+N]**3/3 - xy[i] + I
    for i in range(N):
        for j in range(N):
            duv[i+N] += K[i][j]*(xy[j+N]-xy[i+N])

    return duv

def func_fhn_ring5(t, xy):
    N = 5
    A = 0.7
    B = 0.8
    E = 0.08
    I = 0.32
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if (i-j) in [1, -N+1]:
                K[i][j] = 0.3
            if (i-j) in [-1, N-1]:
                K[i][j] = -0.3
    duv = np.zeros(2*N)
    for i in range(N):
        duv[i] = E*(xy[i+N]+A-B*xy[i])
        duv[i+N] = xy[i+N] - xy[i+N]**3/3 - xy[i] + I
    for i in range(N):
        for j in range(N):
            duv[i+N] += K[i][j]*(xy[j+N]-xy[i+N])

    return duv


K_text = '0.000 0.409 −0.176 −0.064 −0.218 0.464 −0.581 0.101 −0.409 −0.140 \
            0.229 0.000 0.480 −0.404 −0.409 0.040 0.125 0.099 −0.276 −0.131 \
            −0.248 0.291 0.000 −0.509 −0.114 0.429 0.530 0.195 0.416 −0.597 \
            −0.045 0.039 0.345 0.000 0.579 −0.232 0.121 0.130 −0.345 0.463 \
            −0.234 −0.418 −0.195 −0.135 0.000 0.304 0.124 0.038 −0.049 0.183 \
            −0.207 0.536 −0.158 0.533 −0.591 0.000 −0.273 −0.571 0.110 −0.354 \
            0.453 −0.529 −0.287 −0.237 0.470 −0.002 0.000 −0.256 0.438 0.211 \
            −0.050 0.552 0.330 −0.148 −0.326 −0.175 −0.240 0.000 0.263 0.079 \
            0.389 −0.131 0.383 0.413 −0.383 0.532 −0.090 0.025 0.000 0.496 \
            0.459 0.314 −0.121 0.226 0.314 −0.114 −0.450 −0.018 −0.333 0.000'
K_text = K_text.replace('−','-')
K_text = K_text.split()
KR = np.zeros([10, 10])
for i in range(10):
    for j in range(10):
        KR[i][j] = float(K_text[i*10+j])

def func_fhn_random(t, xy):
    N = 10
    A = 0.7
    B = 0.8
    E = 0.08
    #I = 0.32
    I = [0.2]*7+[0.8]*3
    '''
    K_text = '0.000 0.409 −0.176 −0.064 −0.218 0.464 −0.581 0.101 −0.409 −0.140 \
            0.229 0.000 0.480 −0.404 −0.409 0.040 0.125 0.099 −0.276 −0.131 \
            −0.248 0.291 0.000 −0.509 −0.114 0.429 0.530 0.195 0.416 −0.597 \
            −0.045 0.039 0.345 0.000 0.579 −0.232 0.121 0.130 −0.345 0.463 \
            −0.234 −0.418 −0.195 −0.135 0.000 0.304 0.124 0.038 −0.049 0.183 \
            −0.207 0.536 −0.158 0.533 −0.591 0.000 −0.273 −0.571 0.110 −0.354 \
            0.453 −0.529 −0.287 −0.237 0.470 −0.002 0.000 −0.256 0.438 0.211 \
            −0.050 0.552 0.330 −0.148 −0.326 −0.175 −0.240 0.000 0.263 0.079 \
            0.389 −0.131 0.383 0.413 −0.383 0.532 −0.090 0.025 0.000 0.496 \
            0.459 0.314 −0.121 0.226 0.314 −0.114 −0.450 −0.018 −0.333 0.000'
    K_text = K_text.replace('−','-')
    K_text = K_text.split()
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            K[i][j] = float(K_text[i*10+j])
    '''
    K = KR
    duv = np.zeros(2*N)
    for i in range(N):
        duv[i] = E*(xy[i+N]+A-B*xy[i])
        duv[i+N] = xy[i+N] - xy[i+N]**3/3 - xy[i] + I[i]
    for i in range(N):
        for j in range(N):
            duv[i+N] += K[i][j]*(xy[j+N]-xy[i+N])
            #duv[i+N] += K[i][j]*xy[j+N]

    return duv




def make_limitcycle_dataset(model_nm='SL',
                            X0=None, p1=50,
                            num_rotation=3, T=None,
                            dt=0.001,
                            method='RK45',
                            data_interval=1):

    num_rotation = num_rotation
    out = []

    if model_nm == 'SL':
        if T is None:
            T = 2*np.pi/(2*np.pi-1)  # 大よその周期を入れる
        model = StuartLandau
        if X0 is None:
            X0 = make_StuartLandau_inital_state(n=p1)
    if model_nm == 'VP':
        if T is None:
            T = 6.287714285714286
        model = func_vanderPol
    #if model_nm == 'FHN':
    #    if T is None:
    #        T = 3.1149999999999998
    #    model = func_fhn
    if model_nm == 'FHN':
        if T is None:
            T = 36.56
        model = func_fhn3

    if model_nm == 'FHNR':
        if T is None:
            T = 17.66
        model = func_fhn_ring

    if model_nm == 'FHNR3':
        if T is None:
            T = 12.896
        model = func_fhn_ring3

    if model_nm == 'FHNR4':
        if T is None:
            T = 11.25
        model = func_fhn_ring4

    if model_nm == 'FHNR5':
        if T is None:
            T = 12.212
        model = func_fhn_ring5

    if model_nm == 'FHNR6':
        if T is None:
            T = 13.264
        model = func_fhn_ring6

    if model_nm == 'FHNR':
        if T is None:
            T = 17.66
        model = func_fhn_ring
    
    if model_nm == 'FHNRA':
        if T is None:
            T = 75.68884375
        model = func_fhn_random
  
    if model_nm == 'HH':
        if T is None:
            T = 10.127444444444444
        model = func_HodgkinHuxley

    if model_nm == 'WR':
        if T is None:
            T = 2*np.pi/17.25
        model = func_WillamowskiRossler

    if model_nm == 'OS':
        if T is None:
            T = 194.8
        out = 0
        return 

    if dt == (-1):
        dt = T*0.0001


    t = np.arange(0, T*num_rotation, dt)
    for idx, xy in tqdm(enumerate(X0)):
        
        sol = solve_ivp(model, t_span=[t[0], t[-1]],
                        y0=xy, t_eval=t, method=method)
        out.append(sol.y.T[::data_interval])
        #if np.max(np.abs(sol.y.T)) > 300 and model_nm == 'HH':
        #    plt.subplot(1,2,1)
        #    plt.plot(sol.y.T[:,0],sol.y.T[:,1])
        #    plt.subplot(1,2,2)
        #    plt.plot(sol.y.T[:,2],sol.y.T[:,3])
        #    plt.savefig(f'HH_check/{str(idx).zfill(5)}.png')
        #    plt.close()
        #    print(f'{idx},エラーかも')
        #print(f'初期値[{xy}]は上手く計算できませんでした(No{idx},{cnt})')
    out = np.stack(out)
    return out