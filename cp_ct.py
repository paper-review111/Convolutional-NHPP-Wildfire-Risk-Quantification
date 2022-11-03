import torch
import config
import numpy as np
from util import MI_RNN, weighted_adjacency, decay_matrices, network_convolution
import os
import timeit
import matplotlib.pyplot as plt
import pickle


def computation_time_comparison():
    # Compare computational cost between the original model and its inherit RNN
    LN_s = config.LN_s
    M = weighted_adjacency(config.LN_s)
    xi = torch.rand(1)
    beta = torch.rand(5)
    x = torch.rand([len(LN_s), 60, 5])  # set input value

    # collect computation time for MI_RNN with different K
    time_mi_rnn = np.zeros(len(range(1, 21)))
    initial = torch.cat((beta, xi), dim=0)  # initialize with the same beta and xi
    for K in range(1, 21):
        start = timeit.default_timer()
        MI_RNN_0 = MI_RNN(5, initial)  # create a model instance
        prediction, _ = MI_RNN_0(x, None, M, K)
        stop = timeit.default_timer()
        time_mi_rnn[K - 1] = stop - start
        print('Time for mRNN: ', stop - start)

    # collect computation time for the original model with different K (two version)
    x = torch.rand([len(LN_s), 60, 5]).numpy()
    time_lm = np.zeros(len(range(1, 21)))
    for K in range(1, 21):
        start1 = timeit.default_timer()
        x_tilde = network_convolution(x, xi, K)
        x_tilde = torch.from_numpy(x_tilde).type(torch.float)
        loglambda = torch.matmul(x_tilde, beta)
        stop = timeit.default_timer()
        time_lm[K - 1] = stop - start1
        print('Time for cNHPP: ', stop - start1)
    return time_mi_rnn, time_lm


if __name__ == "__main__":
    cwd = os.getcwd()
    result_path = os.path.join(cwd, 'results', 'computational_time_comparison_results.npy')
    computational_time = np.zeros([len(range(1, 21)), 3])
    time_mi_rnn, time_lm = computation_time_comparison()
    computational_time[:, 0] = time_mi_rnn
    computational_time[:, 1] = time_lm
    np.save(result_path, computational_time)  # save md_est_results
    result = np.load(result_path)
    # plot them
    time_lm = result[:, 1]
    time_mi_rnn = result[:, 0]
    K = range(1, 21)
    plt.plot(K, time_lm, marker='|', color='blue', linestyle='-', label='cNHPP')
    plt.plot(K, time_mi_rnn, marker='.', color='red', linestyle='-', label='mRNN')
    plt.legend(fontsize=14)
    plt.ylabel('computation time (s)', fontsize=20)
    plt.xlabel('K', fontsize=20)
    plt.xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], fontsize=14)
    plt.yticks(fontsize=14)
    plot_path = os.path.join(cwd, 'results', 'ct_cp.png')
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
