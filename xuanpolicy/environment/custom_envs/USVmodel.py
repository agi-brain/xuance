"""
Cybership-II USV Dynamic model Model

Input: 
- Force (propeller thurst), Torque (Rudder)

Output: 
- Position pose  = [x, y, psi]
- Velocity vpose = [vx, vy, vpsi]

Disturbance: 
- Wave disturbance model [Yansheng Yang (R.I.P.) 2000]

Parameters:
- Initial Position [x0, y0, psi0]
- Initial Velocity [vx0, vy0, vpsi]
- Wave characteristics [K, phi]

Author: 
- Yuanda Wang (Southeast University; PCL Lab)
- Wenzhang Liu (Anhui University, PCL Lab)

@ Date of Create: Oct. 1, 2021
@ Date of Open-Source: Sep. 27, 2022

"""
import numpy as np

# Cybership-II Physical Parameters
m = 23.8
I_z = 1.76
x_g = 0.046
X_u = -0.7225 
X_uu = -1.3274
X_uuu = -5.8664
X_du = -2.0

Y_r = 0.1079
Y_dr = 0
Y_dv = -10.0
Y_v = -0.8612
Y_vv = -36.2823
Y_rv = -0.805
Y_vr = -0.845
Y_rr = -3.450

N_v = 0.1052
N_dv = 0
N_r = -1.9
N_dr = -1.0

N_vv = 5.0437
N_vr = -0.08
N_rv = 0.130
N_rr = -0.750


M = np.mat(np.array([[m-X_du, 0,          0],
                      [0,      m-Y_dv,     m*x_g-Y_dr],
                      [0,      m*x_g-Y_dr, I_z-N_dr]]))
Minv = M.I

def RK4(ufunc, x0, u, h):
    k1 = ufunc(x0, u)
    k2 = ufunc(x0 + h*k1/2, u)
    k3 = ufunc(x0 + h*k2/2, u)
    k4 = ufunc(x0 + h*k3, u)
    x1 = x0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return x1

class USV_Model():

    def __init__(self, pose0, vel0, wavephi, waveK, dt=0.1):
        self.pose = pose0
        self.vel  = vel0
        self.posvel = vel0
        self.dt = dt
        self.wavephi = wavephi
        self.waveK = waveK
        
    def Cfunc(self):
        vx =  self.vel[0]
        vy =  self.vel[1]
        vphi = self.vel[2]
        
        c11, c12, c21, c22, c33 = 0, 0, 0, 0, 0
        c13 = -(m-Y_dv)*vy - (m*x_g-Y_dr)*vphi
        c23 = (m-X_du)*vx
        c32 = -c23
        c31 = -c13
        
        C = np.array([[c11,c12,c13],[c21,c22,c23],[c31,c32,c33]])
        
        return C
    
    def Dfunc(self):
        vx = np.abs(self.vel[0])
        vy = np.abs(self.vel[1])
        vphi = np.abs(self.vel[2])
        
        d11 = -X_u-X_uu*vx-X_uuu*vx*vx
        d12, d13, d21, d31 = 0, 0, 0, 0
        d22 = -Y_v-Y_vv*vy-Y_rv*vphi
        d23 = -Y_r-Y_vr*vy-Y_rr*vphi
        d32 = -N_v-N_vv*vy-N_rv*vphi
        d33 = -N_r-N_vr*vy-N_rr*vphi
        
        D = np.array([[d11,d12,d13],[d21,d22,d23],[d31,d32,d33]])
        
        return D
    
    def vel_dynamic(self, vpose_dot, u):
        umat = np.mat(u).T
        velmat = np.mat(self.vel).T
        Cmat = np.mat(self.Cfunc())
        Dmat = np.mat(self.Dfunc())
        v_dot = np.array(Minv * (umat - Cmat*velmat - Dmat*velmat))
        v_dot = np.reshape(v_dot, -1)
        
        return v_dot
    
    def pose_dynamic(self, vel, u):
        return u
        
    
    def Jfunc(self):
        phi = self.pose[2]
        
        j11 = np.cos(phi)
        j12 = -np.sin(phi)
        j13 = 0
        j21 = np.sin(phi)
        j22 = np.cos(phi)
        j23 = 0
        j31 = 0
        j32 = 0
        j33 = 1
        J = np.array([[j11,j12,j13],[j21,j22,j23],[j31,j32,j33]])
        
        return J
    
    def wave_force(self, t):
        L = 1.225
        B = 0.29
        d = 0.067
        rho = 1025
        g = 9.81
        w = 2*np.pi*0.1  
        h = 0.2
        k =  2*np.pi/10.0
        
        chi = self.pose[2]-self.wavephi
        
        a = rho*g*(1-np.exp(-k*d))/(k*k)
        b = k*L*np.cos(chi)/2
        c = k*B*np.sin(chi)/2
        
        if np.abs(b) < 0.001:
            F_XW = 0
            F_YW = -2*a*L*np.sin(c)*k*h*np.sin(w*t)/2
            tau_NW = 0
        elif np.abs(c) < 0.001:
            F_XW = 2*a*B*np.sin(b)*k*h*np.sin(w*t)/2
            F_YW = 0
            tau_NW = 0
        else:
            F_XW = 2*a*B*(np.sin(b)*np.sin(c)/c)*k*h*np.sin(w*t)/2
            F_YW = -2*a*L*(np.sin(b)*np.sin(c)/b)*k*h*np.sin(w*t)/2
            c1 = B*B*np.sin(b)*(c*np.cos(c)-np.sin(c))/(c*c)
            c2 = L*L*np.sin(c)*(b*np.cos(b)-np.sin(b))/(b*b)
            tau_NW = a*(c1-c2)*k*h*np.cos(w*t)/2
            
        waveF = self.waveK * np.array([F_XW, F_YW, tau_NW])
        
        return waveF
        
    def step_RK4(self, force, tau, t):
        control = np.array([force, 0, tau])
        wave = self.wave_force(t)
        u = control + wave
        self.vel = RK4(self.vel_dynamic, self.vel, u, self.dt)
        self.dpose = np.array(np.mat(self.Jfunc()) * np.mat(self.vel).T)
        self.dpose = np.reshape(self.dpose, -1)
        self.pose = RK4(self.pose_dynamic, self.pose, self.dpose, self.dt)
        
        return self.pose, self.dpose

if __name__ == '__main__':
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    wavephi = 0
    waveK = 0.1
    usv = USV_Model(pos, vel, wavephi, waveK)
    for t in np.arange(0,10,0.1):
        force = 1
        torque = 0.2
        pose, vel = usv.step_RK4(force, torque, t)
        print('Time: %.1f s' % t)
        print('USV Pos x: %.3f  y: %.3f  psi:%.3f' % (pose[0], pose[1], pose[2]))
        print('USV Vel x: %.3f  y: %.3f  psi:%.3f' % (vel[0], vel[1], vel[2]))
        print('---------------------------------------')
        
        