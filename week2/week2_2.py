import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 7.5), dpi=100)

plt.subplot(4, 1, 1)
x = np.arange(0, 2*np.pi, 0.1)
y_sin = np.sin(x)
plt.plot(x, y_sin, label='sin(x)')
plt.title('Sin')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(4, 1, 2)
j = np.random.rand(10)
k = np.random.rand(10)
plt.scatter(j, k, marker='^', label='k')
plt.title('Scatter')
plt.xlabel('j')
plt.ylabel('k')
plt.legend()

plt.subplot(4, 1, 3)
x = [5,8,10]
y = [12,16,6]
x2 = [6,9,11]
y2 = [6,15,7]
plt.bar(x, y, label='x1')
plt.bar(x2, y2, color='y', label='x2')
plt.title('Bar graph')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


plt.subplot(4, 1, 4)
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
plt.hist(a, bins = [0,20,40,60,80,100], color='r', label='hist')
plt.title("Histogram")
plt.xlabel('bins')
plt.ylabel('hists')
plt.legend()

plt.show()

