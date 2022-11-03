import torch
import config
import numpy as np
from util import MI_RNN, weighted_adjacency, extract_feature
import os
import timeit
import pickle
from datetime import date

cwd = os.getcwd()
result_path = os.path.join(cwd, 'results', 'md_est_conv_results.npy')
result = np.load(result_path)
xi = 0.7
beta = result[1:6]
print(beta)
initial = np.concatenate((beta, xi), axis=None)
initial = torch.from_numpy(initial).type(torch.float)
print(initial)

Model_0 = MI_RNN(5, initial)  # create a model instance
start = date(2019, 5, 25)
end = date(2019, 6, 30)
x = extract_feature(start, end)
x = torch.from_numpy(x).type(torch.float)  # set the input features
fire = config.fire  # input observed points
M = weighted_adjacency(config.LN_s)
optimizer = torch.optim.Adam(Model_0.parameters(), lr=0.001)  # setup optimizer

# traning process
n_steps = 20000
print_every = 1000
size = int(n_steps / print_every)
Loss = []
xi = []
beta0 = []
beta1 = []
beta2 = []
beta3 = []
beta4 = []
for step in range(n_steps):
    prediction, hidden = Model_0(x, None, M, 7)
    loss = MI_RNN.nll_loss(prediction, fire)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    start = timeit.default_timer()
    # display loss and predictions
    if step % print_every == 0:
        print(step)
        Loss.append(loss.item())
        print('Loss:', loss.item())
        print(Model_0.state_dict())
        mi_rnn_xi = Model_0.state_dict().get('MI_RNN.xi')
        xi.append(torch.exp(mi_rnn_xi))
        mi_rnn_beta = Model_0.state_dict().get('MI_RNN.weight').numpy()
        beta0.append(mi_rnn_beta[0])
        beta1.append(mi_rnn_beta[1])
        beta2.append(mi_rnn_beta[2])
        beta3.append(mi_rnn_beta[3])
        beta4.append(mi_rnn_beta[4])
        stop2 = timeit.default_timer()
        print('Time for MLE: ', stop2 - start)

# save the training result
mi_rnn_results_path = os.path.join(cwd, 'results', 'mi_rnn_results.pkl')
mi_rnn_results = [print_every, Loss, xi, beta0, beta1, beta2, beta3, beta4]
with open(mi_rnn_results_path, 'wb') as fp:
    pickle.dump(mi_rnn_results, fp)





