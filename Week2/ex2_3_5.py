#%% create the dataset
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1,11)
tp = np.linspace(0, 1, 100) # more densed grid for findign the polynomial

h = np.array([1.68203, 1.80792, 2.38791,2.67408,2.12245, 2.44969,1.89843, 1.60447,1.80634,1.08810,0.22066])

plt.figure(figsize=(16,4))

# fig 1
plt.subplot(141)
degree = 1
fit = np.polyfit(t, h, degree)
model = np.poly1d(fit)

#plot of observed and modeled data
plt.scatter(t,h, c='b', label='observed')
plt.plot(tp, model(tp),c='r', label='predicted')
plt.xlabel('t [s]')
plt.ylabel('h [m]')
plt.title('Degree = 1')
plt.legend()


# fig 2
plt.subplot(142)
#perform quadratic fit using pylab
degree = 2
fit = np.polyfit(t, h, degree)
model = np.poly1d(fit)

#plot of observed and modeled data
plt.scatter(t,h, c='b', label='observed')
plt.plot(tp, model(tp),c='r', label='predicted')
plt.xlabel('t [s]')
plt.ylabel('h [m]')
plt.title('Degree = 2')
plt.legend()

# fig 3
plt.subplot(143)
degree = 4
fit = np.polyfit(t, h, degree)
model = np.poly1d(fit)

#plot of observed and modeled data
plt.scatter(t,h, c='b', label='observed')
plt.plot(tp, model(tp),c='r', label='predicted')
plt.xlabel('t [s]')
plt.ylabel('h [m]')
plt.title('Degree = 4')
plt.legend()

# fig 3
plt.subplot(144)
#perform higher-degree fit using pylab
degree = 10
fit = np.polyfit(t, h, degree)
model = np.poly1d(fit)

#plot of observed and modeled data
plt.scatter(t,h, c='b', label='observed')
plt.plot(tp, model(tp),c='r', label='predicted')
plt.xlabel('t [s]')
plt.ylabel('h [m]')
plt.title('Degree = 10')
plt.legend()
plt.show()