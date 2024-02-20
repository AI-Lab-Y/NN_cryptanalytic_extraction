'''
Zero-deep NN extraction attack
'''


import numpy as np
import utils


# recover the affine transofrmation of the k-deep NN
# ws, bs are the true nn parameters used to collect decision boundary points
def recover_gamma_p_and_B_p(x, ws, bs, ms=10**(-3), precision=10**(-8)):
    '''
    :param x: a decision boundary point, the array shape is [d_0, 1]
    :param ws: the true weights, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param bs: the biases, the array shape is [layer_num + 1, d_i, 1]
    :param ms: moving stride
    :param precision: the preset precision of the moving stride
    :return: (\gamma_p, B_p), i.e., the parameters of the affine transformation of the k-deep NN
             index, i.e., the subscript of weight used to normalize other weights
    '''
    assert ms > 0
    # start the extraction attack
    # step 2: recover weight signs
    d_0 = len(ws[0][0])
    ws_signs = np.zeros(d_0, dtype=np.int8)
    for j in range(d_0):
        di = np.zeros((d_0, 1), dtype=np.float64)
        di[j][0] = 1.0
        xi_p = x + di * ms
        _, yi_p = utils.f_cheat(x=xi_p, ws=ws, bs=bs)
        xi_n = x - di * ms
        _, yi_n = utils.f_cheat(x=xi_n, ws=ws, bs=bs)
        if yi_p == 1:
            ws_signs[j] = 1
        elif yi_n == 1:
            ws_signs[j] = -1
        else:
            ws_signs[j] = 0

    # step 3: recover weights
    hat_ws = np.zeros((1, d_0), dtype=np.float64)
    hat_bs = np.zeros((1, 1), dtype=np.float64)

    index = d_0 + 1
    for j in range(d_0):
        if ws_signs[j] != 0:
            index = index * 0 + j
            break
    hat_ws[0][index] = ws_signs[index] * 1.0

    di_index = np.zeros((d_0, 1), dtype=np.float64)
    di_index[index][0] = 1.0
    x_index = x + di_index * ms * ws_signs[index]

    for j in range(d_0):
        if ws_signs[j] == 0 or j == index:
            continue
        di_j = np.zeros((d_0, 1), dtype=np.float64)
        di_j[j][0] = 1.0
        suitable_ms_p = utils.binary_search(st=x_index, di=di_j, ws=ws, bs=bs,
                                            ms_low=ms / 1000, ms_high=ms * 1000,
                                            precision=precision)
        suitable_ms_n = utils.binary_search(st=x_index, di=di_j, ws=ws, bs=bs,
                                            ms_low=-1 * ms * 1000, ms_high=-1 * ms / 1000,
                                            precision=precision)
        if suitable_ms_p is not None:
            suitable_ms = suitable_ms_p
        else:
            suitable_ms = suitable_ms_n
            assert suitable_ms_n is not None

        # ws_sign[index] * ms * hat_ws[0][index] + suitable_ms * hat_ws[0][j] = 0
        if suitable_ms == 0:
            print('something is wrong')
            return None
        else:
            hat_ws[0][j] = -1 * ws_signs[index] * ms * hat_ws[0][index] / suitable_ms

    # step 4: recover bias
    hat_bs[0][0] = -1 * np.sum(np.matmul(hat_ws, x))

    return [hat_ws, hat_bs], index


def extract_0_deep_nn(model_path=None, precision=10**(-8)):
    # load model
    fcn = np.load(model_path, allow_pickle=True)
    weights, biases = fcn['arr_0'], fcn['arr_1']

    # start the extraction attack
    # step 1: collect a decision boundary point
    while 1:
        x = utils.find_one_decision_boundary_point(ws=weights, bs=biases,
                                                   layer_num=0, precision=precision)
        if x is not None:
            break

    # hat_ws_bs = [hat_ws, hat_bs]
    hat_ws_bs, index = recover_gamma_p_and_B_p(x=x, ws=weights, bs=biases,
                                               ms=10**(-3), precision=precision)

    # normalize the true weights and bias
    normalized_weights = weights / abs(weights[0][0][index])
    normalized_biases = biases / abs(weights[0][0][index])

    return [[normalized_weights, normalized_biases], hat_ws_bs]


if __name__ == '__main__':
    nn_path = './models/special_and_submit/0_deep_nn_1000.npz'
    # nn_path = './models/special_and_submit/0_deep_nn_10000.npz'

    precision = 10**(-10)
    res = extract_0_deep_nn(model_path=nn_path, precision=precision)
    true_ws, true_bs = res[0][0], res[0][1]
    hat_ws, hat_bs = res[1][0], res[1][1]

    print('normalized true weights with sign are ')
    print('ws: ', true_ws)
    print('bs: ', true_bs)
    print('')
    print('extracted weights are ')
    print('ws: ', hat_ws)
    print('bs: ', hat_bs)

    qw = np.log2(utils.oracle_query_times)
    print('the Oracle query times is 2**{}'.format(qw))

    # compare the prediction matching ratio,
    # using the initial true NN parameters, instead of the normalized one
    fcn = np.load(nn_path, allow_pickle=True)
    true_weights, true_biases = fcn['arr_0'], fcn['arr_1']
    print('start checking the prediction matching ratio')
    prediction_matching_ratio = utils.check_prediction_matching_ratio(t_ws=true_weights,
                                                                      t_bs=true_biases,
                                                                      h_ws=hat_ws,
                                                                      h_bs=hat_bs)
    print('prediction matching ratio is ', prediction_matching_ratio)




