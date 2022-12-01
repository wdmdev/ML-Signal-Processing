import numpy as np

def crosscor(x, y, k):
    x = x[:]
    y = y[:]
    N = min(len(x),len(y))
    
    if (k > N-1) or (k<0):
        raise SystemExit('k should be positive and less than min(Nx,Ny)-1')
    
    kvals = np.arange(-k,k+1).T
    
    r_xy = np.zeros(2*k+1,)
    
    r_xy[k] = np.sum(np.multiply(x[:N], y[:N]))/N
    r_xy[k] = np.sum(x[:N] @ y[:N])/N
    r_xyp = np.zeros(k,)
    r_yxp = np.zeros(k,)

    for i in range(k):
        r_xyp[i]=np.sum(x[i+1:N] @ y[:N-i-1])/N # solution
        r_yxp[i]=np.sum(y[i+1:N] @ x[:N-i-1])/N # solution
    r_xy[:k] = np.flipud(r_yxp)
    r_xy[k+1:2*k+1] = r_xyp
    
    return r_xy, kvals