import numpy as np 

pi = np.pi

################ time integration ############################################
class ND:
    def __init__(self, tmax, dt): # Method called automatically
        self.tmax = tmax
        self.dt = dt
        self.t_ = np.arange(0, tmax, dt)
        self.tnum = len( self.t_ ) # Column length
    
    def evol(self, F, X):
        k1 = self.dt*F(X)
        k2 = self.dt*F(X+0.5*k1)
        k3 = self.dt*F(X+0.5*k2)
        k4 = self.dt*F(X+k3)
        X += (k1+2*k2+2*k3+k4)/6
        return X
        
    def evol_to_convergence(self, F, X):
        Tsimu_num = int(self.tnum)
        for tt in range(Tsimu_num):
            k1 = self.dt*F(X)
            k2 = self.dt*F(X+0.5*k1)
            k3 = self.dt*F(X+0.5*k2)
            k4 = self.dt*F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6 # Runge Kutta
        return X
    
    def find_Tnum_Y(self, F, X, y_basis = 0.0):
        m = 0
        n = 0
        y_temp = X[1,0]
        num = []
        while m < 2:
            k1 = self.dt*F(X)
            k2 = self.dt*F(X+0.5*k1)
            k3 = self.dt*F(X+0.5*k2)
            k4 = self.dt*F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6
            if y_basis < y_temp and y_basis >= X[1,0]: # θ = 0 when the y component falls below y_basis.
                num.append(n)
                if m == 0:
                    Xstart = X # この時のXを初期値とする. 
                m += 1
            n += 1
            if n > 1e8:
                print("Limitcycle doesn't pass 'y_basis'")
                exit
            y_temp = X[1,0]
        Tnum = num[1] - num[0] 
        return Tnum, Xstart
    
    def find_Tnum_X(self, F, X, x_basis):
        m = 0
        n = 0
        x_temp = X[0,0]
        num = []
        while m < 2:
            k1 = self.dt*F(X)
            k2 = self.dt*F(X+0.5*k1)
            k3 = self.dt*F(X+0.5*k2)
            k4 = self.dt*F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6
            if x_basis > x_temp and x_basis <= X[0,0]: # θ = 0 when the x component exceeds x_basis.
                num.append(n)
                if m == 0:
                    Xstart = X # この時のXを初期値とする. 
                m += 1
            n += 1
            if n > 1e8:
                print("Limitcycle doesn't pass 'x_basis'")
                exit
            x_temp = X[0,0]
        Tnum = num[1] - num[0] 
        return Tnum, Xstart

################ Calculate floquet vector (v0, u1, v0, ...)####################################
class Floquet:
    def __init__(self, F, Jacobi, Tnum, T, omega, dt): # Method called automatically
        self.F = F
        self.Jacobi = Jacobi
        self.Tnum = Tnum    
        self.dt = dt
        self.omega = omega
        self.T = T
    
    def Calc_X0_u0(self, X):
        X0_ = np.empty((X.shape[0],self.Tnum))
        u0_ = np.empty((X.shape[0],self.Tnum))
        for tt in range(self.Tnum):
            X0_[:,tt:tt+1] = X
            u0_[:,tt:tt+1] = self.F(X)/self.omega
            k1 = self.dt*self.F(X)
            k2 = self.dt*self.F(X+0.5*k1)
            k3 = self.dt*self.F(X+0.5*k2)
            k4 = self.dt*self.F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6 # Runge Kutta
        return X0_, u0_
    
    def Calc_v0(self, X0_, rotations = 41):
        v0_ = np.empty((X0_.shape[0], X0_.shape[1])) # the same size as X 
        v0_dif = np.empty((X0_.shape[0], X0_.shape[1])) # the same size as X
        v0 = np.ones((X0_.shape[0],1)) # initial point of v0(T)
        
        for rep in range(rotations): # run to convergence
            for tt in range(self.Tnum):
                X = X0_[:,self.Tnum-tt-1:self.Tnum-tt] #list to array
                h = -1/2*self.dt
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6 # X_half_next = X(t-dt/2)
                X_next = np.array([X0_[:,(self.Tnum-tt-2)%self.Tnum]]).T # X_next = X(t-dt)
                
                k1 = self.dt * np.dot(self.Jacobi(X).T, v0)
                k2 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k1/2) )
                k3 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k2/2) )
                k4 = self.dt * np.dot(self.Jacobi(X_next).T, (v0+k3) )
                v0 +=  (k1+2*k2+2*k3+k4)/6 # RungeKutta method
                prob = np.dot(v0.T, self.F(X_next)/self.omega) # production <v0, u0>
            v0 = v0 / prob # normalization every cycle
                
        for tt in range(self.Tnum):#　storage
            X = X0_[:,self.Tnum-tt-1:self.Tnum-tt] # list to array
            v0_[:, self.Tnum-tt-1:self.Tnum-tt] = v0 # storage backwards
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[:,(self.Tnum-tt-2)%self.Tnum]]).T
            
            k1 = self.dt * np.dot(self.Jacobi(X).T, v0)
            k2 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k1/2) )
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T, (v0+k3) )
            v0_next = v0 + (k1+2*k2+2*k3+k4)/6
            v0dif = -(v0_next - v0)/self.dt
            v0_dif[:, self.Tnum-tt-1:self.Tnum-tt] = v0dif        
            v0 = v0_next
        return v0_, v0_dif
    
    def Calc_u1(self, X0_, u0_, v0_, rotations = 2):
        SIZE = u0_.shape[0] 
        u1 = np.ones((SIZE,1)) # initial point of u1(0)
        u1rec = np.empty((SIZE, self.Tnum)) # storage array
        norm1 = np.linalg.norm(u1)
        u1 = u1 / norm1 # normalization

        for rep in range(rotations): # run to convergence
            for tt in range(self.Tnum):
                prod1 = np.dot(v0_[:,tt:tt+1].T, u1) #　<u1, v0>
                u1 = u1 - prod1 * np.array([u0_[:,tt]]).T # remove u0 component
                X = X0_[:,tt:tt+1]

                h = self.dt/2
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6
                X_next = np.array([X0_[:,(tt+1)%self.Tnum]]).T
        
                k1 = np.dot(self.Jacobi(X0_[:,tt:tt+1]),u1) * self.dt
                k2 = np.dot(self.Jacobi(X_half_next), u1 + 0.5*k1) * self.dt
                k3 = np.dot(self.Jacobi(X_half_next), u1 + 0.5*k2) * self.dt
                k4 = np.dot(self.Jacobi(X_next), u1 + k3) * self.dt
                u1 += (k1+2*k2+2*k3+k4)/6
                
            norm1 = np.linalg.norm(u1)
            u1 = u1 / norm1 # normalization
        lambda1 = np.log(norm1) / self.T
        
        for tt in range(self.Tnum): # storage
            u1rec[:,tt:tt+1] = u1
            prod1 = np.dot(np.conjugate(v0_[:,tt:tt+1].T), u1)
            u1 = u1 - prod1 * np.array([u0_[:,tt]]).T 
            X = X0_[:,tt:tt+1]
            ### Runge ###
            h = self.dt/2
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[:,(tt+1)%self.Tnum]]).T
        
            k1 = ( np.dot(self.Jacobi(X0_[:,tt:tt+1]),u1) - lambda1 * u1) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda1*np.identity(SIZE), u1 + k3) * self.dt
        
            u1 += (k1+2*k2+2*k3+k4)/6

        return lambda1, u1rec
    
    def Calc_v1(self, lambda1, X0_, u0_, v0_, u1_, rotations = 2):
        SIZE = v0_.shape[0] 
        v1 = np.ones((SIZE,1)) # initial point of v1(0)
        v1rec = np.empty((SIZE, self.Tnum)) # storage array
        
        for rep in range(rotations):
            for tt in range(self.Tnum): 
                prod1 = np.dot( u0_[:,self.Tnum-tt-1:self.Tnum-tt].T, v1 ) # <v1, u0>
                v1 = v1 - prod1 * np.array([v0_[:,self.Tnum-tt-1]]).T # remove v0 component
                X = X0_[:,self.Tnum-tt-1:self.Tnum-tt] #list to array
                
                h = -1/2*self.dt
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6
                X_next = np.array([X0_[:,(self.Tnum-tt-2)%self.Tnum]]).T
            
                k1 = self.dt * (np.dot(self.Jacobi(X).T, v1) - lambda1 * v1)
                k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k1/2)))
                k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k2/2) )
                k4 = self.dt * np.dot(self.Jacobi(X_next).T - lambda1*np.identity(SIZE), (v1+k3) )
                v1 +=  (k1+2*k2+2*k3+k4)/6
            
            v1 = v1 / np.dot(v1.T, u1_[:,self.Tnum-1:self.Tnum]) # <v1, u1> = 1
        
        for tt in range(self.Tnum): 
            prod1 = np.dot( u0_[:,self.Tnum-tt-1:self.Tnum-tt].T, v1 ) # <v1, u0>
            v1 = v1 - prod1 * np.array([v0_[:,self.Tnum-tt-1]]).T # remove v0 component
            
            v1rec[:,self.Tnum-tt-1:self.Tnum-tt] = v1
            
            X = X0_[:,self.Tnum-tt-1:self.Tnum-tt] #list to array
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[:,(self.Tnum-tt-2)%self.Tnum]]).T
            
            k1 = self.dt * (np.dot(self.Jacobi(X).T, v1) - lambda1 * v1)
            k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k1/2)))
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T - lambda1*np.identity(SIZE), (v1+k3) )
            v1 +=  (k1+2*k2+2*k3+k4)/6
                
        return v1rec
    
    # monodromy matrix
    def monodromy(self, X0_):
        SIZE = X0_.shape[0]  #　state dimension
        M = np.identity(SIZE)  #　M is identity matrix
        for i in range(SIZE):
            y = M[:,i:i+1]  #　y is unit vector
            for tt in range(self.Tnum):  #　evol y for one period. 
                # x(t+dt/2)
                X = X0_[:,tt:tt+1]
                h = 1/2*self.dt
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6
            
                # Runge-Kutta(Jacobi)
                k1 = self.dt*np.dot(self.Jacobi(X), y)
                k2 = self.dt*np.dot(self.Jacobi(X_half_next), y + k1/2)
                k3 = self.dt*np.dot(self.Jacobi(X_half_next), y + k2/2)
                k4 = self.dt*np.dot(self.Jacobi(np.array([X0_[:,(tt+1)%self.Tnum]]).T), y + k3)
                y += (k1+2*k2+2*k3+k4)/6
        return M
    
    def Calc_u1u2(self, lambda1, lambda2, evec1, evec2, X0_, u0_, v0_):
        u1 = np.copy(evec1) # u1(0)
        u2 = np.copy(evec2) # u2(0)
        SIZE = evec1.shape[0]  #　state dimension  
    
        u1rec = np.empty((u1.shape[0], self.Tnum),dtype= complex) #保存用配列
        u2rec = np.empty((u2.shape[0], self.Tnum),dtype= complex) #保存用配列
        
        norm1 = np.linalg.norm(u1)
        norm2 = np.linalg.norm(u2)
        u1 = u1 / norm1 # normalization
        u2 = u2 / norm2
        
        for tt in range(self.Tnum):
            # save
            u1rec[:,tt:tt+1] = u1
            u2rec[:,tt:tt+1] = u2
            
            prod1 = np.dot(np.conjugate(v0_[:,tt:tt+1].T), u1) #　<u1, v0>
            prod2 = np.dot(np.conjugate(v0_[:,tt:tt+1].T), u2) # <u2, v0>
            u1 = u1 - prod1 * np.array([u0_[:,tt]]).T # remove u0 component
            u2 = u2 - prod2 * np.array([u0_[:,tt]]).T 
            
            X = X0_[:,tt:tt+1]
            ### Runge Kutta ###
            h = self.dt/2
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[:,(tt+1)%self.Tnum]]).T
        
            k1 = ( np.dot(self.Jacobi(X0_[:,tt:tt+1]),u1) - lambda1 * u1) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda1*np.identity(SIZE), u1 + k3) * self.dt
            
            u1 += (k1+2*k2+2*k3+k4)/6
            
            k1 = ( np.dot(self.Jacobi(X0_[:,tt:tt+1]),u2) - lambda2 * u2) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda2*np.identity(SIZE), u2 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda2*np.identity(SIZE), u2 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda2*np.identity(SIZE), u2 + k3) * self.dt
        
            u2 += (k1+2*k2+2*k3+k4)/6
        return u1rec, u2rec
    
    def Calc_v1v2(self, lambda1, lambda2, evec1, evec2, X0_, u0_, v0_):
        v1 = np.copy(evec1) # v1(0) = v1(T)
        v2 = np.copy(evec2) # v2(0) = v2(T) 
        SIZE = evec1.shape[0]  #　state dimension  
        
        v1rec = np.empty((v1.shape[0], self.Tnum),dtype= complex) #保存用配列
        v2rec = np.empty((v2.shape[0], self.Tnum),dtype= complex) #保存用配列
        
        # v1,v2を計算
        
        for tt in range(self.Tnum): 
            prod1 = np.dot( np.conjugate(np.array([u0_[:,(self.Tnum-tt)%self.Tnum]])), v1 ) # <u0, v1>
            prod2 = np.dot( np.conjugate(np.array([u0_[:,(self.Tnum-tt)%self.Tnum]])), v2 ) # <u0, v2>
            v1 = v1 - prod1 * np.array([v0_[:,(self.Tnum-tt)%self.Tnum]]).T # remove v0 component
            v2 = v2 - prod2 * np.array([v0_[:,(self.Tnum-tt)%self.Tnum]]).T
            
            X = np.array([X0_[:,(self.Tnum-tt)%self.Tnum]]).T
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = X0_[:,self.Tnum-tt-1:self.Tnum-tt]
            
            k1 = self.dt * (np.dot(self.Jacobi(X).T, v1) - np.conjugate(lambda1) * v1)
            k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - np.conjugate(lambda1)*np.identity(SIZE), (v1+k1/2)))
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - np.conjugate(lambda1)*np.identity(SIZE), (v1+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T - np.conjugate(lambda1)*np.identity(SIZE), (v1+k3) )
            v1 +=  (k1+2*k2+2*k3+k4)/6
            
            k1 = self.dt * (np.dot(self.Jacobi(X).T, v2) - np.conjugate(lambda2) * v2)
            k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - np.conjugate(lambda2)*np.identity(SIZE), (v2+k1/2)))
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - np.conjugate(lambda2)*np.identity(SIZE), (v2+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T - np.conjugate(lambda2)*np.identity(SIZE), (v2+k3) )
            v2 +=  (k1+2*k2+2*k3+k4)/6
            
            # Note that we save v1(t) from (t = T-dt) to (t = 0). 
            # v(T-dt) = v(-dt)
            v1rec[:,self.Tnum-tt-1:self.Tnum-tt] = v1
            v2rec[:,self.Tnum-tt-1:self.Tnum-tt] = v2
            
        return v1rec, v2rec
    
    def product(self, y1_, y2_):
        IP = np.empty((self.Tnum)) # inner product
        for tt in range(self.Tnum):
            y1 = y1_[:,tt:tt+1]
            y2 = y2_[:,tt:tt+1]
            IP[tt] = np.abs(np.dot(np.conjugate(y1).T,y2))
            #IP[tt] = np.dot(np.conjugate(y1).T,y2).real
        return IP

################ Stuart–Landau oscillator ####################################
class SL:
    def __init__(self, a = 2, b = 1):
        self.a = a
        self.b = b
        self.omega = a - b
        
    def dif(self, X):
        x = X[0,0]
        y = X[1,0]
        fx = (x - self.a *y - (x - self.b *y) * (x**2 + y**2))
        fy = (self.a *x + y - (self.b *x + y) * (x**2 + y**2))
        F = np.array([[fx], [fy]])
        return F
    
    def dif_per(self, X, q):
        x = X[0,0]
        y = X[1,0]
        fx = (x - self.a *y - (x - self.b *y) * (x**2 + y**2)) + q[0,0]
        fy = (self.a *x + y - (self.b *x + y) * (x**2 + y**2)) + q[1,0]
        F = np.array([[fx], [fy]])
        return F
    
    def lc(self, t):
        theta = self.omega * t
        X0 = np.array([[np.cos(theta)],
                       [np.sin(theta)]])
        return X0
    
    def lc_theta(self, theta):
        X0 = np.array([[np.cos(theta)],
                       [np.sin(theta)]])
        return X0
    
    def lc_dif(self, t):
        theta = self.omega * t
        X0dif = np.array([[-np.sin(theta)],
                          [np.cos(theta)]])
        return X0dif
    
    def psf(self, t):#phase sensitivity function
        theta = self.omega * t
        Z = np.array([[-self.b * np.cos(theta) - np.sin(theta)],
                       [np.cos(theta) - self.b * np.sin(theta)]])
        return Z
    
    def psf_dif(self, t):
        theta = self.omega * t
        Zdif = np.array([[self.b * np.sin(theta) - np.cos(theta)],
                         [-np.sin(theta) - self.b * np.cos(theta)]])
        return Zdif
    
    def PSF(self, t, Omega):#phase sensitivity function
        theta = Omega * t
        Z = np.array([[-self.b * np.cos(theta) - np.sin(theta)],
                       [np.cos(theta) - self.b * np.sin(theta)]])
        return Z
    
    def PSF_dif(self, t, Omega):
        theta = Omega * t
        Zdif = np.array([[self.b * np.sin(theta) - np.cos(theta)],
                         [-np.sin(theta) - self.b * np.cos(theta)]])
        return Zdif
    
    def psf_theta(self, theta):#phase sensitivity function
        Z = np.array([[-self.b * np.cos(theta) - np.sin(theta)],
                       [np.cos(theta) - self.b * np.sin(theta)]])
        return Z
    
    def psf_dif_theta(self, theta):
        Zdif = np.array([[self.b * np.sin(theta) - np.cos(theta)],
                         [-np.sin(theta) - self.b * np.cos(theta)]])
        return Zdif
    
    def phase(self, X):
        phase = arctan(X[0,0],X[1,0]) - self.b * np.log(np.linalg.norm(X))
        phase = phase % (2*pi)
        return phase
    
    def phi(self, X_, t_):
        phi_ = np.array([[],[]])
        for n, t in enumerate(t_):
            phi = self.phase(X_[0][:,n:n+1]) - self.phase(X_[1][:,n:n+1])
            if pi > abs(phi):  
                phi_ = np.append(phi_, np.array([[t],[phi]]), 1)
            else:
                phi = phi%(2*pi)
                if pi < abs(phi): 
                    phi = phi - 2*pi
                phi_ = np.append(phi_, np.array([[t],[phi]]), 1)
        return phi_

################van der pol#################################################
class VAN:
    # di/dt = self.mu*i - i*i*i/3.0 - v + self.x0
    # dv/dt = i
    def __init__(self, mu, x0 = 1.0, y0 = 0.7):
        self.mu = mu
        self.x0 = x0
        self.y0 = y0
    
    def dif(self, X):
        x = X[0,0]
        y = X[1,0]
        fx = self.mu*x - x*x*x/3.0 - y + self.x0
        fy = x + self.y0
        F = np.array([[fx], [fy]])
        return F
    
    def dif_per(self, X, q):
        x = X[0,0]
        y = X[1,0]
        fx = self.mu*x - x*x*x/3.0 - y + self.x0 + q[0,0]
        fy = x + self.y0 + q[1,0]
        F = np.array([[fx], [fy]])
        return F
    
    def dif_per1(self, X, q1):
        x = X[0,0]
        y = X[1,0]
        fx = self.mu*x - x*x*x/3.0 - y + self.x0 + q1
        fy = x + self.y0
        F = np.array([[fx], [fy]])
        return F
    
    def Jacobian(self, X):
        x = X[0,0]
        f1x = self.mu - x*x
        f1y = -1.0
        f2x = 1.0
        f2y = 0.0
        J = np.array([[f1x, f1y],
                      [f2x, f2y]])
        return J 
    
class VAN_rescale:
    # timescaling  
    def __init__(self, mu, timescale, x0 = 1.0, y0 = 0.7):
        self.mu = mu
        self.x0 = x0
        self.y0 = y0
        self.timescale = timescale
    
    def dif(self, X):
        x = X[0,0]
        y = X[1,0]
        fx = self.timescale * (self.mu*x - x*x*x/3.0 - y + self.x0)
        fy = self.timescale * (x + self.y0)
        F = np.array([[fx], [fy]])
        return F
    
    def dif_per(self, X, q):
        x = X[0,0]
        y = X[1,0]
        fx = self.timescale * (self.mu*x - x*x*x/3.0 - y + self.x0) + q[0,0]
        fy = self.timescale * (x + self.y0) + q[1,0]
        F = np.array([[fx], [fy]])
        return F
    
    def dif_per1(self, X, q):
        x = X[0,0]
        y = X[1,0]
        fx = self.timescale * (self.mu*x - x*x*x/3.0 - y + self.x0) + q
        fy = self.timescale * (x + self.y0)
        F = np.array([[fx], [fy]])
        return F
    
    def Jacobian(self, X):
        x = X[0,0]
        f1x = self.timescale*(self.mu - x*x)
        f1y = -self.timescale
        f2x = self.timescale
        f2y = 0.0
        J = np.array([[f1x, f1y],
                      [f2x, f2y]])
        return J
################ FitHugh-Nagumo oscillator ###################################
class FHN:
    def __init__(self, a = -0.1, b = 0.5, tau = 100, x0 = 1.0, y0 = 0.7):
        #self.Kx = Kx
        self.a = a
        self.b = b
        self.tau = tau
        self.x0 = x0
        self.y0 = y0
    
    def dif(self, X):
        x = X[0,0]
        y = X[1,0]
        fx = x - x*x*x/3.0 - y + self.x0
        fy = (x - self.b*y + self.y0) / self.tau
        F = np.array([[fx], [fy]])
        return F
    
    def dif_per(self, X, q):
        x = X[0,0]
        y = X[1,0]
        fx = x - x*x*x/3.0 - y + self.x0 + q[0,0]
        fy = (x - self.b*y + self.y0) / self.tau + q[1,0]
        F = np.array([[fx], [fy]])
        return F
    
    def dif_per1(self, X, q1):
        x = X[0,0]
        y = X[1,0]
        fx = x - x*x*x/3.0 - y + self.x0 + q1
        fy = (x - self.b*y + self.y0) / self.tau
        F = np.array([[fx], [fy]])
        return F
    
    def Jacobian(self, X):
        x = X[0,0]
        f1x = 1-x*x
        f1y = -1
        f2x = 1 / self.tau 
        f2y = -self.b / self.tau
        J = np.array([[f1x, f1y],
                      [f2x, f2y]])
        return J
    
class HH:
    def __init__(self,)

################ Rossler oscillator ###################################
class Rossler:
    def __init__(self, a = 0.2, b = 0.2, c = 5.7):
        #self.Kx = Kx
        self.a = a
        self.b = b
        self.c = c
    
    def dif(self, X):
        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        fx = - y - z 
        fy = x + self.a*y 
        fz = self.b + x*z - self.c*z 
        F = np.array([[fx], [fy], [fz]])
        return F
    
    def dif_per(self, X, q):
        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        fx = - y - z + q[0,0] 
        fy = x + self.a*y + q[1,0]
        fz = self.b + x*z - self.c*z + q[2,0]
        F = np.array([[fx], [fy], [fz]])
        return F
    
    def Jacobian(self, X):
        #x = X[0,0]
        #y = X[1,0]
        z = X[2,0]
        f1x = 0
        f1y = -1
        f1z = -1
        f2x = 1
        f2y = self.a
        f2z = 0
        f3x = z
        f3y = 0
        f3z = -self.c
        J = np.array([[f1x, f1y, f1z],
                      [f2x, f2y, f2z],
                      [f3x, f3y, f3z]])
        return J
    
################ Willamowski Rossler oscillator ###################################
class WR:
    def __init__(self, b1 = 80, b2 = 20, d1 = 0.16, d2 = 0.13, d3 = 16):
        #self.Kx = Kx
        self.b1 = b1
        self.b2 = b2
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
    
    def dif(self, X):
        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        fx = x*(self.b1 - self.d1*x - y - z) 
        fy = y*(self.b2 - self.d2*y - x) 
        fz = z*(x - self.d3)
        F = np.array([[fx], [fy], [fz]])
        return F
    
    def dif_per(self, X, q):
        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        fx = x*(self.b1 - self.d1*x - y - z) + q[0,0]
        fy = y*(self.b2 - self.d2*y - x) + q[1,0]
        fz = z*(x - self.d3) + q[2,0]
        F = np.array([[fx], [fy], [fz]])
        return F
    
    def Jacobian(self, X):
        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        f1x = self.b1 - 2*self.d1*x - y - z
        f1y = -x
        f1z = -x
        f2x = -y
        f2y = self.b2 - 2*self.d2*y - x
        f2z = 0
        f3x = z
        f3y = 0
        f3z = x - self.d3
        J = np.array([[f1x, f1y, f1z],
                      [f2x, f2y, f2z],
                      [f3x, f3y, f3z]])
        return J

################ arctan #####################################################
def arctan(x, y):
    if x >= 0 and y >= 0:
        theta = np.arctan(y/x)
    elif x < 0 and y >= 0:
        theta = np.arctan(y/x) + pi
    elif x < 0 and y < 0:
        theta = np.arctan(y/x) + pi
    elif x >= 0 and y < 0:
        theta = np.arctan(y/x) + 2*pi
    return theta

### Inner Product #########################################################
def inner_product(y1_, y2_, Tnum):
        IP = np.empty((Tnum))
        for tt in range(Tnum):
            y1 = y1_[:,tt:tt+1]
            y2 = y2_[:,tt:tt+1]
            IP[tt] = np.abs(np.dot(np.conjugate(y1).T,y2))
            #IP[tt] = np.dot(np.conjugate(y1).T,y2).real
        return IP