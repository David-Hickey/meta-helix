#!/usr/bin/env python3

import numpy as np

def force_array(a):
    try:
        len(a)
        return a
    except:
        return np.array([a])

def helix(ts, order : int, ws : list[float], rs : list[float], dt=None):
    if dt is None:
        dt = 10**(-order - 2)

    output = np.zeros(shape=(3, ts.size))

    if order == 0:
        output[2] = ts
    elif order == 1:
        output[0] = np.sin(ts * ws[1]) * rs[1]
        output[1] = np.cos(ts * ws[1]) * rs[1]
        output[2] = ts
    else:
        for i, t in enumerate(force_array(ts)):
            prev_helix = helix(t, order-1, ws, rs)[0]
            prev_helix_dt = helix(t+dt, order-1, ws, rs)[0]
            prev_helix_2dt = helix(t+2*dt, order-1, ws, rs)[0]
        
            T_start = (prev_helix_dt - prev_helix)/dt
            T_end = (prev_helix_2dt - prev_helix_dt)/dt
            
            T_derivative = (T_end - T_start) / dt
            
            basis_vec_1 = T_derivative / np.linalg.norm(T_derivative)
            
            basis_vec_2 = np.cross(T_start.T, T_derivative.T)
            basis_vec_2 /= np.linalg.norm(basis_vec_2)
            
            w = ws[order]
            r = rs[order]                    
            output[:, i] = prev_helix + r*basis_vec_1*np.sin(w*t) + r*basis_vec_2*np.cos(w*t)
            
    return output.T
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    max_t = 6*np.pi/2
    ts = np.linspace(0, max_t, 20000)
    helix_n = helix(ts, 3, [1, 1, 10, 100], [max_t, max_t, max_t*0.1, max_t*0.025])
    
    plt.plot(*helix_n.T)
    plt.show()
