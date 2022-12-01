#%%
"""
This file was translated to python by DTU based on the original 
matlab implementation of Simo Sarkka

Track car state with Kalman filter and Rauch-Tung-Striebel
smoother as in Examples 4.3 and 8.3 of the book:
Simo Sarkka (2013), Bayesian Filtering and Smoothing,
Cambridge University Press. 

This software is distributed under the GNU General Public 
Licence (version 2 or later);
"""
import numpy as np
import matplotlib.pyplot as plt


# Set the parameters

q = 1
dt = 0.1
s = 0.5
A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0,  dt],
    [0, 0, 1,  0],
    [0, 0, 0,  1],
])
Q = q*np.array([
    [dt**3/3, 0,       dt**2/2, 0],
    [0,       dt**3/3, 0,       dt**2/2],
    [dt**2/2, 0,       dt,      0],
    [0,       dt**2/2, 0,       dt],
])

H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])
R = s**2*np.identity(2)
m0 = np.array([[0], [0], [1], [-1]])
P0 = np.identity(4)

# Simulate data

np.random.seed(0)

steps = 100
X = np.zeros((len(A), steps))
Y = np.zeros((len(H), steps))
x = m0
for k in range(steps):
    q = np.linalg.cholesky(Q)@np.random.randn(len(A), 1)
    x = A@x + q
    y = H@x + s*np.random.randn(2, 1)
    X[:, k] = x[:, 0]
    Y[:, k] = y[:, 0]

plt.figure()
plt.plot(X[0, :], X[1, :], '-')
plt.plot(Y[0, :], Y[1, :], '.')
plt.plot(X[0, 0], X[1, 0], 'x')
plt.legend(['Trajectory', 'Measurements'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')


# Kalman filter

m = m0
P = P0
kf_m = np.zeros((len(m), Y.shape[1]))
kf_P = np.zeros((len(P), P.shape[1], Y.shape[1]))
for k in range(Y.shape[1]):
    m = A@m
    P = A@P@A.T + Q

    v = Y[:, k].reshape(-1, 1) - H@m
    S = H@P@H.T + R
    K = P@H.T@np.linalg.inv(S)
    m = m + K@v
    P = P - K@S@K.T

    kf_m[:, k] = m[:, 0]
    kf_P[:, :, k] = P

rmse_raw = np.sqrt(np.mean(np.sum((Y - X[:2, :])**2, 1)))
rmse_kf = np.sqrt(np.mean(np.sum((kf_m[:2, :] - X[:2, :])**2, 1)))

plt.figure()
plt.plot(X[0, :], X[1, :], '-')
plt.plot(Y[0, :], Y[1, :], 'o')
plt.plot(kf_m[0, :], kf_m[1, :], '-')
plt.legend(['True Trajectory', 'Measurements', 'Filter Estimate'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')


# RTS smoother

ms = kf_m[:, -1]
Ps = kf_P[:, :, -1]
rts_m = np.zeros((len(m), Y.shape[1]))
rts_P = np.zeros((len(P), P.shape[1], Y.shape[1]))
rts_m[:, -1] = ms
rts_P[:, :, -1] = Ps
for k in reversed(range(kf_m.shape[1])):
    mp = A@kf_m[:, k]
    Pp = A@kf_P[:, :, k]@A.T+Q
    Gk = kf_P[:, :, k]@A.T@np.linalg.inv(Pp)
    ms = kf_m[:, k] + Gk@(ms - mp)
    Ps = kf_P[:, :, k] + Gk@(Ps - Pp)@Gk.T
    rts_m[:, k] = ms
    rts_P[:, :, k] = Ps

rmse_rts = np.sqrt(np.mean(np.sum((rts_m[:2, :] - X[:2, :])**2, 1)))

plt.figure()
plt.plot(X[0, :], X[1, :], '-')
plt.plot(Y[0, :], Y[1, :], 'o')
plt.plot(rts_m[0, :], rts_m[1, :], '-')
plt.legend(['True Trajectory', 'Measurements', 'Smoother Estimate'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')


plt.show()
