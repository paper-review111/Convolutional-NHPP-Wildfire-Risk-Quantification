import os
import numpy as np
import config
import matplotlib.pyplot as plt
from util import extract_feature, network_convolution, MI_RNN, weighted_adjacency
from datetime import date, timedelta
import pickle
import torch
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------------------------------------------------
# plot the grid search for optimal xi for cNHPP
# -----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd()

result_conv_path = os.path.join(cwd, 'results', 'md_est_conv_results.npy')
result_conv = np.load(result_conv_path)
result_path = os.path.join(cwd, 'results', 'md_est_conv_gird_search.npy')
md_est_results = np.load(result_path)

ll = -md_est_results[:, 0]  # extract log-likehood values
opt = np.argmax(ll)  # find optimal
decays = np.arange(0.1, 1, 0.1)

plt.plot(decays, ll, marker='D', color='black', linestyle='-.')
plt.axvline(x=decays[opt], ymax=1, color='red', linestyle=':')
plt.ylabel('log-likelihood', fontsize=20)
plt.xlabel(r'$\xi$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plot_path = os.path.join(cwd, 'results', 'md_est_plot.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# print the estimated parameters for cNHPP
# -----------------------------------------------------------------------------------------------------------------------
print(
    'for cNHPP, xi:', decays[opt],
    'beta0:', result_conv[1],
    'beta1:', result_conv[2],
    'beta2:', result_conv[3],
    'beta3:', result_conv[4],
    'beta4:', result_conv[5],
    'log-likelihood:', result_conv[0]
)

# -----------------------------------------------------------------------------------------------------------------------
# plot the estimated and predicted wildfire risk for the transmission lines based on cNHPP
# -----------------------------------------------------------------------------------------------------------------------
start = date(2019, 6, 1)
end = date(2019, 7, 7)
days = (end - start).days + 1

result_conv_path = os.path.join(cwd, 'results', 'md_est_conv_results.npy')
result_conv = np.load(result_conv_path)

x_conv = extract_feature(start - timedelta(7), end)
x_tilde_conv = network_convolution(x_conv, 0.7, 7)
conv_beta = result_conv[1:6]
conv_lambda = np.exp(np.matmul(x_tilde_conv, conv_beta))

days_plot = [0, 14, 29, 30, 32, 34]
LN = config.LN
for day in days_plot:
    fig, ax = plt.subplots(1, figsize=(10, 10))
    plt.xlabel('long', fontsize=26, color='black')
    plt.ylabel('lat', fontsize=26, color='black')
    if day >= 30:
        plt.title(label='2019-7-' + str(day - 29), fontsize=24)
    else:
        plt.title(label='2019-6-' + str(day + 1), fontsize=24)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    LN.plot(ax=ax, column=np.array(conv_lambda[:, day]), legend=True, linewidth=3, cmap='plasma')
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=18)
    filename = 'est_risk' + str(day) + '.png'
    plot_path = os.path.join(cwd, 'results', filename)
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# plot the convergence of the mRNN model
# -----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd()
mi_rnn_results_path = os.path.join(cwd, 'results', 'mi_rnn_results.pkl')

with open(mi_rnn_results_path, 'rb') as fp:
    mi_rnn_results = pickle.load(fp)

Loss = mi_rnn_results[1]
xi = mi_rnn_results[2]
beta0 = mi_rnn_results[3]
beta1 = mi_rnn_results[4]
beta2 = mi_rnn_results[5]
beta3 = mi_rnn_results[6]
beta4 = mi_rnn_results[7]
bins = len(Loss)

print_every = mi_rnn_results[0]
epochs = np.arange(bins) * print_every

plt.plot(epochs, Loss, marker='*', color='black', linestyle='-.')
plt.xticks([0, 5000, 10000, 15000, 20000])
plt.ylabel('-log-likelihood', fontsize=20)
plt.xlabel('epochs', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plot_path = os.path.join(cwd, 'results', 'mi_rnn_ll.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

plt.plot(epochs, xi, marker='*', color='black', linestyle='-.')
plt.xticks([0, 5000, 10000, 15000, 20000])
plt.xlabel('epochs', fontsize=20)
plt.ylabel(r'$\xi$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plot_path = os.path.join(cwd, 'results', 'mi_rnn_xi.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

plt.plot(epochs, beta0, marker='*', color='black', linestyle='-.')
plt.xticks([0, 5000, 10000, 15000, 20000])
plt.xlabel('epochs', fontsize=20)
plt.ylabel(r'$\beta_0$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plot_path = os.path.join(cwd, 'results', 'mi_rnn_beta0.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

plt.plot(epochs, beta1, marker='*', color='black', linestyle='-.')
plt.xticks([0, 5000, 10000, 15000, 20000])
plt.xlabel('epochs', fontsize=20)
plt.ylabel(r'$\beta_1$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plot_path = os.path.join(cwd, 'results', 'mi_rnn_beta1.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

plt.plot(epochs, beta2, marker='*', color='black', linestyle='-.')
plt.xticks([0, 5000, 10000, 15000, 20000])
plt.xlabel('epochs', fontsize=20)
plt.ylabel(r'$\beta_2$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plot_path = os.path.join(cwd, 'results', 'mi_rnn_beta2.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

plt.plot(epochs, beta3, marker='*', color='black', linestyle='-.')
plt.xticks([0, 5000, 10000, 15000, 20000])
plt.xlabel('epochs', fontsize=20)
plt.ylabel(r'$\beta_3$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plot_path = os.path.join(cwd, 'results', 'mi_rnn_beta3.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

plt.plot(epochs, beta4, marker='*', color='black', linestyle='-.')
plt.xticks([0, 5000, 10000, 15000, 20000])
plt.xlabel('epochs', fontsize=20)
plt.ylabel(r'$\beta_4$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plot_path = os.path.join(cwd, 'results', 'mi_rnn_beta4.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# print the estimated parameters from different models
# -----------------------------------------------------------------------------------------------------------------------
start = date(2019, 6, 1)
end = date(2019, 6, 30)
days = (end - start).days + 1
# HPP
lambda_ = len(config.fire) / (len(config.LN_s) * days)
ll = len(config.fire) * np.log(lambda_) - days * len(config.LN_s) * lambda_
print('for HPP, lambda:', lambda_, 'log-likelihood for HPP:', -ll)
# NHPP
result_var_path = os.path.join(cwd, 'results', 'md_est_var_results.npy')
result_var = np.load(result_var_path)
print(
    'for NHPP, beta0:', result_var[1],
    'beta1:', result_var[2],
    'beta2:', result_var[3],
    'beta3:', result_var[4],
    'beta4:', result_var[5],
    'log-likelihood:', result_var[0]
)
# mRNN
print(
    'for mRNN, xi:', xi[len(xi)-1].numpy()[0],
    'beta0:', beta0[len(beta0)-1],
    'beta1:', beta1[len(beta1)-1],
    'beta2:', beta2[len(beta2)-1],
    'beta3:', beta3[len(beta3)-1],
    'beta4:', beta4[len(beta4)-1],
    'log-likelihood:', -Loss[len(Loss)-1]
)

# -----------------------------------------------------------------------------------------------------------------------
# plot the estimated wildfire intensity distribution by different models
# -----------------------------------------------------------------------------------------------------------------------
# HPP model
HPP_lambda = np.full((len(config.LN), days), lambda_)
# NHPP
x = extract_feature(start, end)
var_beta = result_var[1:6]
var_lambda = np.exp(np.matmul(x, var_beta))
# mi-RNN
last = len(xi) - 1
xi = xi[last]
beta0 = beta0[last]
beta1 = beta1[last]
beta2 = beta2[last]
beta3 = beta3[last]
beta4 = beta4[last]
initial = torch.tensor([beta0, beta1, beta2, beta3, beta4, xi])
Model_0 = MI_RNN(5, initial)  # create a model instance
x_mi_rnn = extract_feature(start - timedelta(7), end)
x_mi_rnn = torch.from_numpy(x_mi_rnn).type(torch.float)  # set the input features
M = weighted_adjacency(config.LN_s)
prediction, hidden = Model_0(x_mi_rnn, None, M, 7)
mi_rnn_lambda = np.exp(prediction.detach().numpy())
# cNHPP
x_conv = extract_feature(start - timedelta(7), end)
x_tilde_conv = network_convolution(x_conv, 0.7, 7)
conv_beta = result_conv[1:6]
conv_lambda = np.exp(np.matmul(x_tilde_conv, conv_beta))
# plot at the selected days
for i in [0, 14, 29]:
    df = {
        'HPP': HPP_lambda[:, i] * 1e5,
        'var-NHPP': var_lambda[:, i] * 1e5,
        'mi-RNN': mi_rnn_lambda[:, i] * 1e5,
        'conv-NHPP': conv_lambda[:, i] * 1e5
    }
    sns.set_theme(style="white", palette=None, font_scale=2)
    sns.kdeplot(df['var-NHPP'], shade=False, color="black", label='NHPP', linewidth=1.5)
    sns.kdeplot(df['mi-RNN'], shade=False, color="b", label='mRNN', linewidth=1.5)
    sns.kdeplot(df['conv-NHPP'], shade=False, color="r", label='cNHPP', linewidth=1.5)
    plt.axvline(x=7.815, ymax=1, color='green', linestyle=':', linewidth=3)
    plt.legend()
    plt.xlabel(r'$\hat{\lambda}(\times10^{-5})$')
    filename = 'est_intensity_cp' + str(i) + '.png'
    plot_path = os.path.join(cwd, 'results', filename)
    plt.title(label='2019-6-' + str(i + 1))
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# plot the percentiles of the estimated intensities associated with the power lines with fires
# -----------------------------------------------------------------------------------------------------------------------
conv_percentile = []
for i in range(len(config.fire)):
    day = config.fire.Day[i]
    line = config.fire.LN_id[i]
    ecdf = ECDF(conv_lambda[:, day])
    conv_percentile.append(ecdf(conv_lambda[line, day]))

var_percentile = []
for i in range(len(config.fire)):
    day = config.fire.Day[i]
    line = config.fire.LN_id[i]
    ecdf = ECDF(var_lambda[:, day])
    var_percentile.append(ecdf(var_lambda[line, day]))

mi_percentile = []
for i in range(len(config.fire)):
    day = config.fire.Day[i]
    line = config.fire.LN_id[i]
    ecdf = ECDF(mi_rnn_lambda[:, day])
    mi_percentile.append(ecdf(mi_rnn_lambda[line, day]))

barWidth = 0.2
br1 = np.arange(15)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig, ax = plt.subplots(1, figsize=(25, 10))
plt.bar(br1, var_percentile, label='NHPP', width=barWidth)
plt.bar(br2, mi_percentile, label='mRNN', width=barWidth)
plt.bar(br3, conv_percentile, label='cNHPP', width=barWidth)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
plt.xlabel('wildfire', fontsize=34)
plt.ylabel('percentile', fontsize=34)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xticks(np.arange(15) + barWidth,
           ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])
plt.legend(fontsize=24)
plot_path = os.path.join(cwd, 'results', 'percentile')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
