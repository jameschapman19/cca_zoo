import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")

directory = 'Results/1000_runs_sparse_final/sparse_run_0/'

all_w_weights = np.load(directory + 'w_weights.npy')
for i in range(all_w_weights.shape[1]):
    plt.scatter(np.arange(all_w_weights.shape[2]), np.abs(all_w_weights[0, i, :] / np.abs(all_w_weights[0, i, :]).max()).T,
                label='Dimension ' + str(i))
    # plt.scatter(np.arange(100),np.abs(all_weights[0,1,:]).T,label='Dimension 1')
    # plt.scatter(np.arange(100),np.abs(all_weights[0,2,:]).T,label='Dimension 2')
    plt.legend()
    plt.title('PLS dimension ' + str(i) + ' brain variable weights')
    plt.xlabel('Principal component #')
    plt.ylabel('Absolute weight as a percentage of maximum')
    plt.savefig(directory+'PLS_dimension_' + str(i) + 'brain_weights')
    plt.show()

for i in range(all_w_weights.shape[1]):
    plt.scatter(np.arange(all_w_weights.shape[2]), np.abs(all_w_weights[4, i, :] / np.abs(all_w_weights[4, i, :]).max()).T,
                label='Dimension ' + str(i))
    # plt.scatter(np.arange(100),np.abs(all_weights[0,1,:]).T,label='Dimension 1')
    # plt.scatter(np.arange(100),np.abs(all_weights[0,2,:]).T,label='Dimension 2')
    plt.legend()
    plt.title('SCCA dimension ' + str(i) + ' brain variable weights')
    plt.xlabel('Principal component #')
    plt.ylabel('Absolute weight as a percentage of maximum')
    plt.savefig(directory+'SCCA_dimension_' + str(i) + 'brain_weights')
    plt.show()

for i in range(all_w_weights.shape[1]):
    plt.scatter(np.arange(all_w_weights.shape[2]), np.abs(all_w_weights[5, i, :] / np.abs(all_w_weights[5, i, :]).max()).T,
                label='Dimension ' + str(i))
    # plt.scatter(np.arange(100),np.abs(all_weights[0,1,:]).T,label='Dimension 1')
    # plt.scatter(np.arange(100),np.abs(all_weights[0,2,:]).T,label='Dimension 2')
    plt.legend()
    plt.title('Generalized SCCA dimension ' + str(i) + ' brain variable weights')
    plt.xlabel('Principal component #')
    plt.ylabel('Absolute weight as a percentage of maximum')
    plt.savefig(directory+'GSCCA_dimension_' + str(i) + 'brain_weights')
    plt.show()

all_c_weights = np.load(directory + 'c_weights.npy')
for i in range(all_w_weights.shape[1]):
    plt.scatter(np.arange(all_c_weights.shape[2]), np.abs(all_c_weights[0, i, :] / np.abs(all_c_weights[0, i, :]).max()).T,
                label='Dimension ' + str(i))
    # plt.scatter(np.arange(100),np.abs(all_weights[0,1,:]).T,label='Dimension 1')
    # plt.scatter(np.arange(100),np.abs(all_weights[0,2,:]).T,label='Dimension 2')
    plt.legend()
    plt.title('PLS dimension ' + str(i) + ' behaviour variable weights')
    plt.xlabel('Principal component #')
    plt.ylabel('Absolute weight as a percentage of maximum')
    plt.savefig(directory+'PLS_dimension_' + str(i) + 'behaviour_weights')
    plt.show()

for i in range(all_w_weights.shape[1]):
    plt.scatter(np.arange(all_c_weights.shape[2]), np.abs(all_c_weights[4, i, :] / np.abs(all_c_weights[4, i, :]).max()).T,
                label='Dimension ' + str(i))
    # plt.scatter(np.arange(100),np.abs(all_weights[0,1,:]).T,label='Dimension 1')
    # plt.scatter(np.arange(100),np.abs(all_weights[0,2,:]).T,label='Dimension 2')
    plt.legend()
    plt.title('SCCA dimension ' + str(i) + ' behaviour variable weights')
    plt.xlabel('Principal component #')
    plt.ylabel('Absolute weight as a percentage of maximum')
    plt.savefig(directory+'SCCA_dimension_' + str(i) + 'behaviour_weights')
    plt.show()

for i in range(all_w_weights.shape[1]):
    plt.scatter(np.arange(all_c_weights.shape[2]), np.abs(all_c_weights[5, i, :] / np.abs(all_c_weights[5, i, :]).max()).T,
                label='Dimension ' + str(i))
    # plt.scatter(np.arange(100),np.abs(all_weights[0,1,:]).T,label='Dimension 1')
    # plt.scatter(np.arange(100),np.abs(all_weights[0,2,:]).T,label='Dimension 2')
    plt.legend()
    plt.title('Generalized SCCA dimension ' + str(i) + ' behaviour variable weights')
    plt.xlabel('Principal component #')
    plt.ylabel('Absolute weight as a percentage of maximum')
    plt.savefig(directory+'GSCCA_dimension_' + str(i) + 'behaviour_weights')
    plt.show()

print('here')
