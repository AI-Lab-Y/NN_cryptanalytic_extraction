import numpy as np
from copy import deepcopy

oracle_query_times = 0


def f_cheat(x, ws, bs, count=1):
    '''
    :param x: the input
    :param ws: the weights of each layer, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param bs: the biases of each layer, the array shape is [layer_num + 1, d_i, 1]
    :param count: if count = 1, increase the counter.
    :return: the hard label of the input
    '''
    if count:
        global oracle_query_times
        oracle_query_times += 1

    layer_num = len(ws)
    h = x
    for i in range(layer_num):
        h = np.matmul(ws[i], h) + bs[i]
        if i != layer_num - 1:
            h = h * (h > 0)
    assert len(h) == 1
    soft_label = np.squeeze(h)

    # return the hard-label
    if soft_label > 0:
        hard_label = 1
    else:
        hard_label = 0
    return soft_label, hard_label


def get_map(x, ws, bs):
    def convert_array_into_scalar(arr):
        n = len(arr)
        res = 0
        for i in range(n):
            res += arr[n-1-i] * (2**i)
        return res

    map = []
    layer_num = len(ws)
    h = x
    for i in range(layer_num):
        h = np.matmul(ws[i], h) + bs[i]
        if i != layer_num - 1:
            # get the model activation pattern
            tp = np.squeeze(h > 0)
            tp = convert_array_into_scalar(tp)
            map.append(tp)

            h = h * (h > 0)
    return map


def binary_search(st, di, ws, bs, ms_low=0, ms_high=2**20, precision=10**(-8)):
    '''
    :param st: the starting point, the array shape is [d_0, 1]
    :param di: the moving direction, the array shape is [d_0, 1]
    :param ws: the weights of each layer, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param bs: the biases of each layer, the array shape is [layer_num + 1, d_i, 1]
    :return: the suitable moving stride that makes the st reach the decision boundary
    '''
    _, y_ref = f_cheat(x=st, ws=ws, bs=bs)
    _, y_low = f_cheat(x=st + di * ms_low, ws=ws, bs=bs)
    _, y_high = f_cheat(x=st + di * ms_high, ws=ws, bs=bs)

    if y_low != y_high:
        while abs(ms_high - ms_low) > precision:
            ms_mid = ms_low + (ms_high - ms_low) / 2
            x_mid = st + di * ms_mid
            _, y_mid = f_cheat(x=x_mid, ws=ws, bs=bs)
            if y_mid == y_low:
                if ms_low == ms_mid:
                    break
                else:
                    ms_low = ms_mid
            else:
                if ms_high == ms_mid:
                    break
                else:
                    ms_high = ms_mid
        return ms_low + (ms_high - ms_low) / 2
    else:
        return None


def find_one_decision_boundary_point(ws, bs, layer_num=0, precision=10**(-8)):
    '''
    :param ws: the weights of each layer, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param bs: the biases of each layer, the array shape is [layer_num + 1, d_i, 1]
    :param layer_num: the number of hidden layers, i.e., k
    :param precision: the preset precision of the moving stride
    :return: collected decision boundary points, the array shape is [point_num, d_0, 1]
    '''
    assert len(ws) == layer_num + 1
    assert len(bs) == layer_num + 1
    d_0 = len(ws[0][0])

    st = np.zeros((d_0, 1), dtype=np.float64)
    di = np.random.normal(loc=0.0, scale=1.0, size=(d_0, 1))
    ms_p = binary_search(st=st, di=di, ws=ws, bs=bs,
                         ms_low=2 ** (-10), ms_high=2 ** 20, precision=precision)
    ms_n = binary_search(st=st, di=di, ws=ws, bs=bs,
                         ms_low=-2 ** 20, ms_high=-2 ** (-10), precision=precision)
    if ms_p is not None:
        point = st + di * ms_p
    elif ms_n is not None:
        point = st + di * ms_n
    else:
        point = None

    return point


# when checking the prediction matching ratio, we do not count the Oracle query times.
# Because we can test PMR by reusing queries.
def check_prediction_matching_ratio(t_ws, t_bs, h_ws, h_bs, n=10**6):
    '''
    :param t_ws: true weights, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param t_bs: true biases, the array shape is [layer_num + 1, d_i, 1]
    :param h_ws: extracted weights, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param h_bs: extracted biases, the array shape is [layer_num + 1, d_i, 1]
    :return: the prediction matching ratio
    '''
    d_0 = t_ws[0].shape[1]

    res = 0
    for i in range(n):
        x = np.random.normal(loc=0.0, scale=1.0, size=(d_0, 1))
        _, yt = f_cheat(x=x, ws=t_ws, bs=t_bs, count=0)
        _, yh = f_cheat(x=x, ws=h_ws, bs=h_bs, count=0)
        if yt == yh:
            res += 1
    return res / n


def filter_duplicate_vectors(arr, l1_error=10**(-3)):
    '''
    :param arr: vectors to be processed
    :param l1_error: the error threshold for recognizing duplicate vectors
    :return: the array of indexs of non-duplicate vectors
    '''
    n = len(arr)
    res = []
    res.append(0)
    m = 1
    for i in range(1, n):
        flag = 1
        for j in range(m):
            tmp = arr[i] - arr[res[j]]
            if np.max(np.abs(tmp)) < l1_error:
                flag = flag ^ 1
                break
        if flag == 1:
            res.append(i)
            m += 1
    return res


def align_deep_nns(ws_real=None, bs_real=None, ws_extract=None, bs_extract=None, nn_shape=None):
    print('ws_real is ', ws_real)

    err_bound = 10**(-2)

    # create a copy of ws_real
    aligned_ws_real = deepcopy(ws_real)
    aligned_bs_real = deepcopy(bs_real)

    def compute_norm_factors(i):
        numerator_factors = np.ones((d_0, 1), dtype=np.float64)
        denominator_factors = ws_real[0][:, 0].reshape(-1, 1)
        if i == 1:
            return numerator_factors, denominator_factors
        else:
            numerator_factors_new = deepcopy(denominator_factors)
            for j in range(1, i - 1):
                numerator_factors_new = np.matmul(ws_real[j], numerator_factors_new)
            if i - 1 > 0:
                denominator_factors = np.matmul(ws_real[i-1], numerator_factors_new)
            return numerator_factors_new, denominator_factors

    # create an identity matrix
    d_0 = nn_shape[0]

    layer_num = len(nn_shape)
    for i in range(1, layer_num):
        # print('i is ', i)
        d_i = nn_shape[i]

        # compute the numerator_factors and denominator_factors
        numerator_factors, denominator_factors = compute_norm_factors(i)
        # print('numerator is ', numerator_factors)
        # print('denomicator is ', denominator_factors)
        assert denominator_factors.shape[1] == 1
        assert denominator_factors.shape[0] == d_i
        # normalize current layer weights and biases
        for j in range(d_i):
            aligned_ws_real[i-1][j] = aligned_ws_real[i-1][j] / abs(denominator_factors[j][0])
            aligned_bs_real[i-1][j] = aligned_bs_real[i-1][j] / abs(denominator_factors[j][0])
        d_i_minus_1 = nn_shape[i-1]
        for j in range(d_i_minus_1):
            aligned_ws_real[i-1][:, j] = aligned_ws_real[i-1][:, j] * abs(numerator_factors[j][0])

        print('after normalization, ws are: ')
        print(aligned_ws_real[i-1])
        print(ws_extract[i-1])

        # adjust the neuron order of current layer of the victim nn
        index = []
        for j_e in range(d_i):
            for j_r in range(d_i):
                if np.max(aligned_ws_real[i-1][j_r] - ws_extract[i-1][j_e]) < err_bound:
                    index.append(j_r)
                    break

        # adjust weights (rows) and biases of current layer according to the new neuron order
        tmp_ws = deepcopy(aligned_ws_real[i-1])
        tmp_bs = deepcopy(aligned_bs_real[i-1])
        print('index is ', index)
        for j_r in range(d_i):
            aligned_ws_real[i-1][j_r] = tmp_ws[index[j_r]]
            aligned_bs_real[i-1][j_r] = tmp_bs[index[j_r]]

        # update ws_real according to the neuron order
        tmp_ws = deepcopy(ws_real[i-1])
        for j_r in range(d_i):
            ws_real[i-1][j_r] = tmp_ws[index[j_r]]

        # adjust weights (columns) of the next layer according to the new neuron order
        if i != layer_num - 1:
            tmp_ws = deepcopy(aligned_ws_real[i])
            for j_r in range(d_i):
                aligned_ws_real[i][:, j_r] = tmp_ws[:, index[j_r]]

        # update ws_real according to the neuron order
        if i != layer_num - 1:
            tmp_ws = deepcopy(ws_real[i])
            for j_r in range(d_i):
                ws_real[i][:, j_r] = tmp_ws[:, index[j_r]]

    return aligned_ws_real, aligned_bs_real