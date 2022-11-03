import os
from datetime import date
import numpy as np
from scipy.optimize import minimize
from util import LogLikFunc, network_convolution, extract_feature
import timeit
import matplotlib.pyplot as plt


start = date(2019, 5, 25)
end = date(2019, 6, 30)
x = extract_feature(start, end)


def training(input):
    start = timeit.default_timer()
    beta0 = np.array([0.1, 0, 0, 0, 0])
    result = minimize(LogLikFunc, beta0, args=input, method='L-BFGS-B')
    stop = timeit.default_timer()
    print('Time for MLE: ', stop - start)
    print(result.fun)
    print(result.x)
    return result


if __name__ == "__main__":
    # --------------------------------------------------------------------------------------------------------
    # for NHPP training
    # --------------------------------------------------------------------------------------------------------
    input = x[:, 7:100, :]
    print(input.shape)
    result = training(input)
    cwd = os.getcwd()
    result_path = os.path.join(cwd, 'results', 'md_est_var_results.npy')
    md_est_results = np.zeros(6)  # first: func value, others: \bm{beta}
    md_est_results[0] = result.fun
    md_est_results[1:6] = result.x
    np.save(result_path, md_est_results)  # save md_est_results
    # --------------------------------------------------------------------------------------------------------
    # for cNHPP training
    # --------------------------------------------------------------------------------------------------------
    # search the optimal xi
    decays = np.arange(0.1, 1, 0.1)
    md_est_results = np.zeros([len(decays),  6])  # first: func value, others: \bm{beta}
    for i in range(len(decays)):
        decay = decays[i]
        print(x.shape)
        start = timeit.default_timer()
        input = network_convolution(x, decay, 7)[:, 0:100, :]
        print(input.shape)
        stop = timeit.default_timer()
        print('Time for calculate inputs: ', stop - start)
        result = training(input)
        md_est_results[i, 0] = result.fun
        md_est_results[i, 1:6] = result.x
    cwd = os.getcwd()
    result_path = os.path.join(cwd, 'results', 'md_est_conv_gird_search.npy')
    np.save(result_path, md_est_results)
    opt = np.argmin(md_est_results[:, 0])
    # optimize other parameters in cNHPP
    decay = decays[opt]
    print(x.shape)
    start = timeit.default_timer()
    input = network_convolution(x, decay, 7)[:, 0:100, :]
    print(input.shape)
    stop = timeit.default_timer()
    print('Time for calculate inputs: ', stop - start)
    result = training(input)
    cwd = os.getcwd()
    result_path = os.path.join(cwd, 'results', 'md_est_conv_results.npy')
    md_est_results = np.zeros(6)  # first: func value, others: \bm{beta}
    md_est_results[0] = result.fun
    md_est_results[1:6] = result.x
    np.save(result_path, md_est_results)  # save md_est_results
