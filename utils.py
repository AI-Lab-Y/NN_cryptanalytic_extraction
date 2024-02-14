import numpy as np

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
        while abs(ms_high - ms_low) >= precision:
            ms_mid = ms_low + (ms_high - ms_low) / 2
            x_mid = st + di * ms_mid
            _, y_mid = f_cheat(x=x_mid, ws=ws, bs=bs)
            if y_mid == y_low:
                if ms_mid != ms_low:
                    ms_low = ms_mid
                else:
                    ms_low = ms_mid - precision * 0.5
            else:
                if ms_mid != ms_high:
                    ms_high = ms_mid
                else:
                    ms_high = ms_mid + precision * 0.5
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
def check_prediction_matching_ratio(t_ws, t_bs, h_ws, h_bs):
    '''
    :param t_ws: true weights, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param t_bs: true biases, the array shape is [layer_num + 1, d_i, 1]
    :param h_ws: extracted weights, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param h_bs: extracted biases, the array shape is [layer_num + 1, d_i, 1]
    :return: the prediction matching ratio
    '''
    d_0 = t_ws[0].shape[1]

    n = 10**6
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