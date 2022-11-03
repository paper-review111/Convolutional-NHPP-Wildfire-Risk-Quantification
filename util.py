import gemgis as gg
from shapely.geometry import LineString, Point, Polygon, MultiLineString
import numpy as np
import datetime as dt
import config
import torch
from torch import nn
import warnings
import os
from datetime import datetime

warnings.simplefilter("ignore")


class ModelLayer(nn.Module):
    def __init__(self, size, initial):
        super(ModelLayer, self).__init__()
        self.size = size
        self.initial = initial
        weight = torch.Tensor(self.size)
        self.weight = nn.Parameter(weight)
        xi = torch.Tensor(1)
        self.xi = nn.Parameter(xi)
        # set initial wight and xi from its original model to fast optimization
        for i in range(size):
            nn.init.constant_(self.weight[i], initial[i])
        nn.init.constant_(self.xi, np.log(initial[size]))

    def forward(self, x, hidden, M, k):
        M = torch.exp(self.xi) * M
        tempt = torch.matmul(x, self.weight)
        output = torch.zeros(tempt.shape)
        if hidden is None:
            hidden = torch.zeros(tempt.shape[0])
        for i in range(tempt.shape[1]):
            dummy = torch.matmul(M, hidden) + tempt[:, i]
            output[:, i] = dummy
            hidden = dummy
        return output[:, k:x.shape[1]], hidden


class MI_RNN(nn.Module):
    def __init__(self, size, initial):
        super(MI_RNN, self).__init__()
        self.initial = initial
        self.size = size
        self.MI_RNN = ModelLayer(size, initial)

    def forward(self, x, hidden, M, k):
        return self.MI_RNN(x, hidden, M, k)

    @staticmethod
    def nll_loss(output, fire):
        rv = torch.sum(output[fire.LN_id, fire.Day]) - torch.sum(torch.exp(output))
        return -rv


def weighted_adjacency(LN_s):
    M = torch.zeros(len(LN_s), len(LN_s))
    for i in range(len(LN_s)):
        NBor = np.array(LN_s.NBor[i])
        NBorWeight = np.ones(len(NBor)) / NBor.size
        NBorWeight_tensor = torch.from_numpy(NBorWeight).type(torch.float)
        M[i, NBor] = NBorWeight_tensor
    return M


def decay_matrices(M, xi, K):
    matrices = []
    for j in range(K):
        if j == 0:
            tempt = xi * M
            matrices.append(tempt)
        else:
            tempt = xi * torch.matmul(M, tempt)
            matrices.append(tempt)
    return matrices


def network_convolution(x, decay, k):
    M = weighted_adjacency(config.LN_s)
    matrices = decay_matrices(M, decay, k)
    x_tilde = torch.zeros([x.shape[0], x.shape[1] - k, x.shape[2]])
    for t in range(x.shape[1] - k):
        tempt = torch.zeros([x.shape[0], x.shape[2]])
        for i in range(k):
            x_tempt = torch.from_numpy(x[:, t + (k - (i + 1)), :]).type(torch.float)
            tempt += torch.matmul(matrices[i], x_tempt)
        x_tilde[:, t, :] = tempt + x[:, t + k, :]
    return x_tilde.numpy()


def LineCutFunc(ls):
    """
     Cut the EIA transmission line into a linear network with segments
    """
    seg = []
    if type(ls) == LineString:
        tempt_list = gg.vector.explode_linestring_to_elements(linestring=ls)
        n_tempt = len(tempt_list)
        for j in range(n_tempt):
            seg.append(tempt_list[j])
    else:
        n_linestring = len(ls.geoms)
        for m in range(n_linestring):
            tempt_list = gg.vector.explode_linestring_to_elements(linestring=ls.geoms[m])
            n_tempt = len(tempt_list)
            for j in range(n_tempt):
                seg.append(tempt_list[j])
    return seg


def DayCalFunc(date, starting_date):
    """
     Calculate the days starting from a date
    """
    start = dt.datetime.strptime(starting_date, '%Y-%m-%d')
    now = dt.datetime.strptime(date, '%Y-%m-%d')
    return (now - start).days


def SegLogLikFunc(i, coef, featureConv):
    """
     Loglikelihood function for a line segment
    """
    fire = config.fire
    term1 = 0
    fire_id = np.where(fire['LN_id'].values == i)[0]
    for j in fire_id:
        term1 += np.dot(coef, featureConv[i, fire.Day[j],])
    term2 = 0
    for t in range(np.shape(featureConv)[1]):
        term2 += np.exp(np.dot(coef, featureConv[i, t,]))
    return term1 - term2


def LogLikFunc(coef, featureConv):
    """
     Loglikelihood function for all segments
    """
    LN = config.LN  # set global variable
    loglik = 0
    for i in range(len(LN)):
        loglik += SegLogLikFunc(i, coef, featureConv)
    return -loglik


def extract_feature(start, end):
    # load feature data
    cwd = os.getcwd()
    modis_ft_path = os.path.join(cwd, 'data', 'ndvi.npy')  # load features from the nasa_ndvi
    hrrr_ft_path = os.path.join(cwd, 'data', 'hrrr.npy')  # load features from the noaa-hrrr
    modis_ft = np.load(modis_ft_path)
    hrrr_ft = np.load(hrrr_ft_path)

    # obtain the scaling factor
    start2 = datetime(2019, 5, 25)
    end2 = datetime(2019, 6, 30)
    days2 = (end2 - start2).days + 1
    modis_ft_dim = np.shape(modis_ft)
    hrrr_ft_dim = np.shape(hrrr_ft)
    ftn_nasa = modis_ft_dim[2]  # number of retrieved features from the nasa modis product
    ftn_noaa = hrrr_ft_dim[2]  # number of retrieved features from the noaa hrrr model
    y = np.zeros((modis_ft_dim[0], days2, ftn_noaa + ftn_nasa + 1))
    y[:, :, 0] = np.ones((modis_ft_dim[0], days2))
    y[:, :, 1:(ftn_nasa + 1)] = modis_ft[:, 0:days2, :]
    y[:, :, (ftn_nasa + 1):(ftn_nasa + ftn_noaa + 2)] = hrrr_ft[:, 0:days2, :]

    # create feature array from noaa and nasa
    days = (end - start).days + 1
    x = np.zeros((modis_ft_dim[0], days, ftn_noaa + ftn_nasa + 1))
    x[:, :, 0] = np.ones((modis_ft_dim[0], days))
    x[:, :, 1:(ftn_nasa + 1)] = modis_ft[:, 0:days, :]
    x[:, :, (ftn_nasa + 1):(ftn_nasa + ftn_noaa + 2)] = hrrr_ft[:, 0:days, :]
    for covariate in range(ftn_noaa + ftn_nasa + 1):
        x[:, :, covariate] = x[:, :, covariate] / np.max(y[:, :, covariate])  # scale these covariates
    return x
