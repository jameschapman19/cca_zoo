import matplotlib

matplotlib.use('agg')
import numpy as np
from sklearn import metrics
import pylab
import scipy.cluster.hierarchy as spc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

"""
A bunch of methods I have added to help me do plotting when needed

cv_plot() used to automatically generate basic hyperparameter plots for linear wrapper cv_fit() method

plot_results() used to generate comparison plots for HCP data


"""


def cv_plot(scores, param_dict, reg):
    # First see if 1 dimensional
    plt.figure()
    dims = isinstance(next(iter(param_dict.values()))[0], tuple)
    if dims == 0:
        x1_name = list(param_dict.keys())[0]
        x1_vals = list(param_dict.values())[0]
        lineObjects = plt.plot(np.array(x1_vals), scores)
    elif len(param_dict) == 1:
        parameter_name = list(param_dict.keys())[0]
        x1_vals = set([pair[0] if (isinstance(pair, tuple)) else pair for pair in list(param_dict.values())[0]])
        x2_vals = set([pair[1] if (isinstance(pair, tuple)) else pair for pair in list(param_dict.values())[0]])
        x1_name = parameter_name + '_1'
        x2_name = parameter_name + '_2'
        lineObjects = plt.plot(np.array(sorted(list(x1_vals))),
                               np.squeeze(scores.reshape((len(x1_vals), len(x2_vals), -1)).mean(axis=-1)))
        plt.legend(lineObjects, sorted(list(x2_vals)), title=x2_name)
    else:
        x1_name = list(param_dict.keys())[0]
        x1_vals = list(param_dict.values())[0]
        x2_name = list(param_dict.keys())[1]
        x2_vals = list(param_dict.values())[1]
        lineObjects = plt.plot(np.array(sorted(list(x1_vals))),
                               np.squeeze(scores.reshape((len(x1_vals), len(x2_vals), -1)).mean(axis=-1)))
        plt.legend(lineObjects, sorted(list(x2_vals)), title=x2_name)

    plt.xlabel(x1_name)
    plt.title('Hyperparameter plot ' + reg)
    plt.ylabel('Score (sum of first n correlations)')
    plt.savefig('Hyperparameter_plot ' + reg)


def plot_results(data, labels):
    # data is c*3*k where c is the different models and k is the number of latents and 3 is train,test,val

    # Compare sum of first k dimensions
    corr_sum = np.sum(data, axis=2)

    # set width of bar
    barWidth = 0.7
    r = 2 * np.arange(len(labels))
    r1 = [x - barWidth / 2 for x in r]
    r2 = [x + barWidth / 2 for x in r]

    # Make the plot
    fig, ax = plt.subplots()
    ax.bar(r1, corr_sum[:, 0], width=barWidth, edgecolor='white', label='Train')
    ax.bar(r2, corr_sum[:, 1], width=barWidth, edgecolor='white', label='Test')

    # Add xticks on the middle of the group bars
    ax.set_xlabel('model', fontweight='bold')
    ax.set_ylabel('Sum of first n correlations', fontweight='bold')
    # plt.xticks([r + barWidth for r in range(len(labels))], labels)
    ax.set_xticks(r)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(rotation=90)

    # Create legend & Show graphic
    ax.legend()
    fig.tight_layout()
    fig.savefig('compare_train_test')

    # Train dimensions
    plt.figure()
    x = np.arange(1, data.shape[2] + 1)
    for i, m in enumerate(labels):
        if any(m in _ for _ in ['KCCA', 'KCCA-reg', 'KCCA-gaussian', 'KCCA-polynomial', 'DCCA']):
            plt.plot(x, data[i, 0, :], linestyle='dashed')
        else:
            plt.plot(x, data[i, 0, :])
    plt.title('train canonical correlations')
    plt.legend(labels)
    plt.xlabel('Dimension')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.savefig('train_dims')

    # Test dimensions
    plt.figure()
    for i, m in enumerate(labels):
        if any(m in _ for _ in ['KCCA', 'KCCA-reg', 'KCCA-gaussian', 'KCCA-polynomial', 'DCCA']):
            plt.plot(x, data[i, 1, :], linestyle='dashed')
        else:
            plt.plot(x, data[i, 1, :])
    plt.title('test canonical correlations')
    plt.legend(labels)
    plt.xlabel('Dimension')
    plt.ylabel('Correlation')
    plt.tight_layout()
    plt.savefig('test_dims')


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1 / odds]) * 100


def plot_distributions(y_true, Z_true, y_pred, Z_pred=None, epoch=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    subplot_df = (
        Z_true
            .assign(race=lambda x: x['race'].map({1: 'white', 0: 'black'}))
            .assign(sex=lambda x: x['sex'].map({1: 'male', 0: 'female'}))
            .assign(y_pred=y_pred)
    )
    _subplot(subplot_df, 'race', ax=axes[0])
    _subplot(subplot_df, 'sex', ax=axes[1])
    _performance_text(fig, y_true, Z_true, y_pred, Z_pred, epoch)
    fig.tight_layout()
    return fig


def _subplot(subplot_df, col, ax):
    for label, df in subplot_df.groupby(col):
        sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)
    ax.set_title(f'Sensitive attribute: {col}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_yticks([])
    ax.set_ylabel('Prediction distribution')
    ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(col))


def _performance_text(fig, y_test, Z_test, y_pred, Z_pred=None, epoch=None):
    if epoch is not None:
        fig.text(1.0, 0.9, f"Training epoch #{epoch}", fontsize='16')

    clf_roc_auc = metrics.roc_auc_score(y_test, y_pred)
    clf_accuracy = metrics.accuracy_score(y_test, y_pred > 0.5) * 100
    p_rules = {'race': p_rule(y_pred, Z_test['race']),
               'sex': p_rule(y_pred, Z_test['sex']), }
    fig.text(1.0, 0.65, '\n'.join(["Classifier performance:",
                                   f"- ROC AUC: {clf_roc_auc:.2f}",
                                   f"- Accuracy: {clf_accuracy:.1f}"]),
             fontsize='16')
    fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
                                 [f"- {attr}: {p_rules[attr]:.0f}%-rule"
                                  for attr in p_rules.keys()]),
             fontsize='16')
    if Z_pred is not None:
        adv_roc_auc = metrics.roc_auc_score(Z_test, Z_pred)
        fig.text(1.0, 0.20, '\n'.join(["Adversary performance:",
                                       f"- ROC AUC: {adv_roc_auc:.2f}"]),
                 fontsize='16')


def plot_weights(w, c):
    # Train dimensions
    fig, ax = plt.subplots()
    ax.plot(w[:, 0], label='brain weights', color="blue")
    ax.set_xlabel('PCA component')
    ax.set_ylabel('CCA_archive weights across input PCA components (brain)')
    ax2 = ax.twinx()
    ax2.plot(c[:, 0], label='behaviour weights', color="red")
    ax2.set_ylabel('CCA_archive weights across input PCA components (behaviour)')
    plt.tight_layout()
    plt.savefig('weight')


def plot_connectome_correlations(ordered_connectivity, cca_connectivity, linkage):
    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0.01, 0.7, 0.3, 0.3])
    Z1 = spc.dendrogram(linkage)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.01, 0.4, 0.3, 0.3])
    im = axmatrix.matshow(ordered_connectivity, aspect='auto', cmap=pylab.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.01, 0.1, 0.3, 0.3])
    im2 = axmatrix.matshow(cca_connectivity, aspect='auto', cmap=pylab.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    fig.savefig('connectivity_correlation.png')


def plot_latent_space(z_x, z_y, conf=None, conf_labels=None):
    # Maybe we can make a confounds more opaque at some point
    assert (z_x.shape == z_y.shape)
    outdims = z_x.shape[1]
    if conf is not None:
        outdims += 1
    fig, axs = plt.subplots(1, outdims)
    for p in range(outdims - 1):
        axs[p].plot(z_x[:, p], z_y[:, p], 'o')
        slope, intercept, r_value, p_value, std_err = linregress(z_x[:, p], z_y[:, p])
        axs[p].plot(np.unique(z_x[:, p]), slope * np.unique(z_x[:, p]) + intercept)
        axs[p].text(0, 0, '$R^2 = %0.2f$' % r_value ** 2)
        axs[p].set_title('Dimension: ' + str(p))
    if conf is not None:
        axs[outdims - 1].bar(np.arange(conf.shape[1]), conf.mean(axis=0))
        axs[outdims - 1].axhline(conf.mean(axis=0).mean(), color='blue', linewidth=2)
    fig.suptitle('Plot of Latent Space')


def plot_training_loss(train, val):
    plt.figure()
    plt.plot(train, label='Train')
    plt.plot(val, label='Val')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative correlation sum)')
    plt.legend()
    plt.savefig('training_loss')
