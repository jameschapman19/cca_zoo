import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

N = 1000
xData = np.random.normal(0, 50, N)
yData = np.random.normal(0, 10, N)
xData = np.reshape(xData, (N, 1))
yData = np.reshape(yData, (N, 1))
data = np.hstack((xData, yData))
eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
data=data@eigenvectors
x_rot = xData*np.cos(np.pi/4.)-yData*np.sin(np.pi/4.)
y_rot = xData*np.sin(np.pi/4.)+yData*np.cos(np.pi/4.)

rot_data = np.hstack((x_rot, y_rot))

mu = data.mean(axis=0)
mu_rot = rot_data.mean(axis=0)
data = data - mu
data_rot = rot_data - mu_rot
# data = (data - mu)/data.std(axis=0)  # Uncommenting this reproduces mlab.PCA results
eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
eigenvectors_rot, eigenvalues_rot, V_rot = np.linalg.svd(data_rot.T, full_matrices=False)
projected_data = np.dot(data, eigenvectors)
sigma=[]
sigma.append(-projected_data.std(axis=0).mean())
sigma.append(projected_data.std(axis=1).mean())
projected_data_rot = np.dot(data_rot, eigenvectors_rot)
sigma_rot=[]
sigma_rot.append(-projected_data_rot.std(axis=0).mean())
sigma_rot.append(projected_data_rot.std(axis=1).mean())
print(eigenvectors)

fig, axs = plt.subplots(ncols=2,sharex=True,sharey=True)

axs[0].scatter(x_rot, y_rot)
for i,axis in enumerate(eigenvectors_rot):
    start, end = mu_rot, mu_rot + 2 * sigma_rot[i] * axis
    axs[0].annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='black', width=2.0))
axs[0].set_xlabel('Original X1')
axs[0].set_ylabel('Original Y1')
axs[0].set_aspect('equal')
axs[0].set_title('original data')
axs[0].set_xlim(-200,200)
axs[0].set_ylim(-200,200)

axs[1].scatter(data[:,0],data[:,1])
for i,axis in enumerate(eigenvectors):
    start, end = mu, mu + 2 * sigma[i] * axis
    axs[1].annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='black', width=2.0))
axs[1].set_xlabel('PCA X1')
axs[1].set_ylabel('PCA Y1')
axs[1].set_aspect('equal')
axs[1].set_title('Principal Components')

fig.suptitle('Principal Components Analysis for Generated Data',size=16)
plt.tight_layout()
plt.savefig('PCA')
plt.show()