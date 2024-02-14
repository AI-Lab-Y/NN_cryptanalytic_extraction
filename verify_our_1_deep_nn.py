'''
One-deep NN extraction attack
'''


import numpy as np
from itertools import combinations
import utils

total_query_times = 0


def expected_nn_parameters(nn_shape, ws):
    print('when the recovered model is correct: ')
    print('expected w_1s are: ')
    for i in range(nn_shape[1]):
        print(ws[0][i] / abs(ws[0][i][0]))
    print('expected w_2 is: ')
    expected_w_2 = np.array([ws[1][0][i] * abs(ws[0][i][0]) for i in range(nn_shape[1])])
    tp = np.array([ws[1][0][i] * ws[0][i][0] for i in range(nn_shape[1])])
    expected_w_2 = expected_w_2 / abs(np.sum(tp))
    print(expected_w_2)

    print('when we regard k-deep nn as a zero-deep nn: ')
    print('expected gamma_ps are: ')
    for i in range(nn_shape[1]):
        print(ws[0][i] * ws[1][0][i] / abs(ws[0][i][0] * ws[1][0][i]))


# recover the affine transofrmation of the k-deep NN
# ws, bs are the true nn parameters used to collect decision boundary points
def recover_gamma_p(x, ws, bs, ms=10**(-3), precision=10**(-8)):
    '''
    :param x: a decision boundary point, the array shape is [d_0, 1]
    :param ws: the true weights, the array shape is [layer_num + 1, d_i, d_{i-1}]
    :param bs: the biases, the array shape is [layer_num + 1, d_i, 1]
    :param precision: the preset precision of the moving stride
    :return: \gamma_p, i.e., the weights of the affine transformation of the k-deep NN
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
    # assert ws_signs[0] != 0
    if ws_signs[0] == 0:
        return None

    # step 3: recover weights
    hat_ws = np.zeros((1, d_0), dtype=np.float64)
    hat_ws[0][0] = ws_signs[0] * 1.0

    # let x move to x_index, such that z(f(x_index)) == 1
    di_0 = np.zeros((d_0, 1), dtype=np.float64)
    di_0[0][0] = 1.0
    x_index = x + di_0 * ms * ws_signs[0]
    _, y_index = utils.f_cheat(x=x_index, ws=ws, bs=bs)
    assert y_index == 1

    for j in range(1, d_0):
        if ws_signs[j] == 0:
            continue
        di_j = np.zeros((d_0, 1), dtype=np.float64)
        di_j[j][0] = 1.0
        suitable_ms_p = utils.binary_search(st=x_index, di=di_j, ws=ws, bs=bs,
                                            ms_low=ms / 1000, ms_high=ms * 100,
                                            precision=precision)
        suitable_ms_n = utils.binary_search(st=x_index, di=di_j, ws=ws, bs=bs,
                                            ms_low=-1 * ms * 100, ms_high=-1 * ms / 1000,
                                            precision=precision)
        if suitable_ms_p is not None:
            suitable_ms = suitable_ms_p
        elif suitable_ms_n is not None:
            suitable_ms = suitable_ms_n
        else:
            return None

        # ws_signs[0] * ms * hat_ws[0][index] + suitable_ms * hat_ws[0][j] = 0
        if suitable_ms == 0:
            print('something is wrong')
            return None
        else:
            hat_ws[0][j] = -1 * ws_signs[0] * ms * hat_ws[0][0] / suitable_ms

    return hat_ws


# recover the weight vector of the j-th neuron in layer i, by solving a system of linear equations
def recover_ws_via_soe(di_s, extracted_ws, target_ws, layer_no, selected_indexs):
    '''
    :param di_s: the list consisting of the dimension in each layer
    :param extracted_ws: extracted weights in layers 1,...,i-1, the array shape is [i-1, d_j, d_{j-1}]
    :param target_ws: the weight vector (\gamma_P) of the affine transformation, shape is [1, d_0]
    :param layer_no: the number of the attacked layer, i.e., i
    :return: extracted weight vector of the target neuron in layer i
    '''
    assert layer_no - 1 == len(extracted_ws) == 1
    d_0 = di_s[0]
    d_i_minus_1 = di_s[layer_no - 1]
    assert d_0 >= d_i_minus_1

    # build the coefficient array
    h = extracted_ws[layer_no - 2]
    for i in range(layer_no - 3, -1, -1):
        h = np.matmul(h, extracted_ws[i])
    h = h.T     # array shape is [d_0, d_i_minus_1]

    # choose d_i_minus_1 out of d_0  coefficient array
    CM = []
    for i in range(len(selected_indexs)):
        CM.append(h[selected_indexs[i]])

    # build the value array
    target_ws = np.squeeze(target_ws)
    y = []
    for i in range(len(selected_indexs)):
        y.append(target_ws[selected_indexs[i]])

    # print('coefficient matrix is ', h)
    # print('value array is ', y)

    # solving the system of linear equations
    soln, *rest = np.linalg.lstsq(CM, y, 1e-6)
    # reshape soln into [1, d_i_minus_1]
    ans = np.array([soln])

    return ans


# recover all the biases by solving a system of linear equations
def recover_bs_via_soe(di_s, B_P, group_index, extracted_ws):
    '''
    :param di_s: the list consisting of the dimension in each layer
    :param B_P: recovered n biases B_p of f(x) = \gamma_p x + B_p, the array shape is (n, 1)
    :param extracted_ws: extracted weights, the array shape is [k+1, d_i, d_{i-1}]
    :return:
    '''
    layer_num = len(di_s) - 1
    assert len(extracted_ws) == layer_num == 2

    # neuron num
    n = np.sum(np.array(di_s)[1:layer_num])

    # build the coefficients of [b_1^1,...,b_{d_1}^1, b_1^2,...,b_{d_2}^2, ...  , b_1^{k+1}]
    h = np.zeros((n+1, n+1), dtype=np.float64)

    # consider the n decision boundary points,
    # not including the one that makes all the neurons active
    for i in range(n):
        h[i][i] = extracted_ws[1][0][i]
        h[i][n] = 1

    for j in range(n):
        h[n][j] = extracted_ws[1][0][j]
    h[n][n] = 1

    # build the value array
    y = B_P

    # solving the system of linear equations
    soln, *rest = np.linalg.lstsq(h, y, 1e-6)

    # reshape soln, [k+1, d_i, 1]
    bs = []
    cnt = 0
    for i in range(layer_num):
        b_i = []
        di = di_s[i+1]
        for j in range(di):
            b_i.append(soln[cnt])
            cnt += 1
        b_i = np.array(b_i, dtype=np.float64).reshape((-1, 1))
        bs.append(b_i)
        # bs.append(np.array(b_i, dtype=np.float64))

    return bs


def compare_model_signature(di_s, gamma_ps, extratced_ws, l1_error=10**(-3), confidence=0.6):
    hidden_layer_num = len(di_s) - 2
    n = np.sum(np.array(di_s)[1:hidden_layer_num+1])

    def convert_scalar_to_arrays(p):
        res = []
        cnt = 0
        for i in range(hidden_layer_num, 0, -1):
            di = di_s[i]
            tp = np.zeros(di, dtype=np.float64)
            for j in range(di):
                v = (p >> cnt) & 1
                tp[di - 1 - j] = v * 1.0
                cnt += 1
            res.append(tp)
        ans = [res[i-1] for i in range(hidden_layer_num, 0, -1)]
        return ans

    # we do not check the model activation pattern with p^{(i)} = 0
    def check_validity(p):
        flag = 1
        for i in range(len(p)):
            if np.sum(p[i]) == 0:
                flag = flag ^ 1
                break
        return flag

    extracted_gamma_ps = []

    # collect the gamma_ps under the extracted nn parameters
    total_num = 2**n
    for p in range(total_num):
        DMs = [np.zeros((di_s[i+1], di_s[i+1]), dtype=np.float64) for i in range(hidden_layer_num)]
        ps = convert_scalar_to_arrays(p)
        if check_validity(ps) == 0:
            continue

        # compute the diagonal matrix
        for i in range(hidden_layer_num):
            di = di_s[i+1]
            for j in range(di):
                if ps[i][j] == 1:
                    DMs[i][j][j] = 1

        # compute the gamma_p
        h = extratced_ws[hidden_layer_num]
        for i in range(hidden_layer_num, 0, -1):
            h = np.matmul(h, DMs[i-1])
            h = np.matmul(h, extratced_ws[i-1])
        assert h.shape[0] == 1 and h.shape[1] == di_s[0]

        # ignore the gamma_p with h[0][0] = 0 temporarily
        if h[0][0] != 0:
            extracted_gamma_ps.append(h / abs(h[0][0]))
    ps_num = len(extracted_gamma_ps)

    # chech whether the true gamma_ps is a subset of the collected gamma_ps
    gamma_ps_num = len(gamma_ps)
    cnt = 0
    for i in range(gamma_ps_num):       # the true model signature
        flag = 0
        for j in range(ps_num):         # the extracted model signature
            diff_vec = gamma_ps[i] - extracted_gamma_ps[j]
            if np.max(np.abs(diff_vec)) < l1_error:
                # print('model activation pattern is ', j + 1)
                # print('diff_vector is ', diff_vec)
                flag = flag ^ 1
                break
        cnt += flag
    if (cnt * 1.0 / gamma_ps_num) >= confidence:
        return 1
    else:
        return 0


def get_dynamic_confidence(complete_gamma_p, unique_indexs, l1_error=10**(-3)):
    gamma_num = len(complete_gamma_p)
    unique_index_num = len(unique_indexs)
    cnt = np.zeros(unique_index_num, dtype=np.uint32)
    for i in range(unique_index_num):
        index = unique_indexs[i]
        for j in range(gamma_num):
            if j in unique_indexs:
                cnt[i] += 1
                continue
            if np.max(np.abs(complete_gamma_p[j] -complete_gamma_p[index])) < l1_error:
                cnt[i] += 1
    suitable_confidence = 1 - np.sum(cnt < 2) / unique_index_num
    return suitable_confidence


def extract_1_deep_nn(model_path=None, nn_shape=None, precision=10**(-8), l1_error=10**(-3)):
    # load model
    fcn = np.load(model_path, allow_pickle=True)
    weights, biases = fcn['arr_0'], fcn['arr_1']

    print('true weights are ')
    print(weights)
    print('true biases are ')
    print(biases)

    expected_nn_parameters(nn_shape=nn_shape, ws=weights)

    # the number of neurons
    neuron_num = 0
    for i in range(1, len(nn_shape) - 1):
        neuron_num += nn_shape[i]
    n = 8 * (2**neuron_num)

    # start the extraction attack
    # steps 1: collect decision boundary points
    print('start step 1: collect decision boundary points')
    tmp_boundary_points = []
    tmp_maps = []
    cnt = 0
    while cnt < n:
        # x is a decision boundary point
        x = utils.find_one_decision_boundary_point(ws=weights, bs=biases,
                                                   layer_num=1, precision=precision)
        if x is not None:
            cur_maps = utils.get_map(x=x, ws=weights, bs=biases)
            tmp_maps.append(cur_maps)
            tmp_boundary_points.append(x)

            soft_label, _ = utils.f_cheat(x=x, ws=weights, bs=biases)
            # print('boundary point: ', cnt)
            # print('soft label is ', soft_label)
            # print('model activation pattern is ', cur_maps[0])
        cnt += 1
    tmp_boundary_point_num = len(tmp_boundary_points)

    if tmp_boundary_point_num == 0:
        print('the created NN is special, we do not collect any decision boundary points')
        return

    # -----------------------------------------------------------------------------------------------

    # step 2: recover \gamma_p
    print('start step 2: recover $\gamma_P$s')
    valid_boundary_points = []
    gamma_p = []
    maps = []
    for i in range(tmp_boundary_point_num):
        x = tmp_boundary_points[i]
        # hat_ws, hat_ws shape: (1, d_0)
        hat_ws = recover_gamma_p(x=x, ws=weights, bs=biases,
                                 ms=10**(-3), precision=precision)
        if hat_ws is not None:
            valid_boundary_points.append(x)
            gamma_p.append(hat_ws)
            maps.append(tmp_maps[i])
            # print('valid boundary point: ', i)
            # print('extracted weights are ', hat_ws)

    # filter decision boundary points with duplicate model activation patterns
    # whether two MAPs are the same are verified by their parameters \gamma_p of the affine transformation
    if len(gamma_p) != 0:
        unique_indexs = utils.filter_duplicate_vectors(arr=gamma_p, l1_error=l1_error)
        print('{} different MAPs occur'.format(len(unique_indexs)))

        print('corresponding \gamma_P are: ')
        for i in unique_indexs:
            print('boundary point: {}, \gamma_p is {}'.format(i, gamma_p[i]))

        print(len(unique_indexs), ' true MAPs are ', [maps[j] for j in unique_indexs])
    else:
        unique_indexs = []
        print('0 valid decision boundary points')

    # filter 0: the number of unique MAPs should exceed the number of neurons.
    if len(unique_indexs) < neuron_num + 1:
        print('filtered by filter 1 (do not collect enough unique MAPs)')
        return

    print('We have collected all the required queries')
    qw = np.log2(utils.oracle_query_times)
    print('the Oracle query times is 2**{}'.format(qw))

    print('start recover nn under a subset of decision boundary points')

    model_num = 0
    model_candidate_num = 0
    # select d_1 + 1 decision boundary points each time
    m = nn_shape[1]
    # first select d_1 decision boundary points that only makes the i-th neuron active in layer 1
    for group_index in combinations(unique_indexs, m):
        for w_2_sign in [1, -1]:
            # step 3: recover weights w_1 in layer 1
            # print('start step 3')

            hat_w_1 = [w_2_sign * gamma_p[cur_i][0] for cur_i in group_index]
            hat_w_1 = np.array(hat_w_1, dtype=np.float64)

            # select 1 decision boundary point that makes all the $d_1$ neurons active
            for i in unique_indexs:
                if i in group_index:
                    continue

                # increase the counter of tested model
                model_num += 1

                selected_maps = [maps[v] for v in group_index]
                selected_maps.append(maps[i])

                # step 4: recover weights w_2 in layer 2
                # print('start step 4')
                cur_hat_ws = [hat_w_1]
                hat_w_2 = recover_ws_via_soe(di_s=nn_shape,
                                             extracted_ws=np.array(cur_hat_ws, dtype=np.float64),
                                             target_ws=gamma_p[i], layer_no=2,
                                             selected_indexs=[j for j in range(nn_shape[1])])

                # filter 2: the signs of recovered w_2 should be the same as the expectation
                if np.any(hat_w_2 * w_2_sign < 0):
                    # print('filtered by filter 2')
                    continue

                # filter 3: check whether the model signature of extracted nn is equivalent to the true one
                dynamic_confidence = get_dynamic_confidence(complete_gamma_p=gamma_p,
                                                            unique_indexs=unique_indexs,
                                                            l1_error=l1_error)
                # print('dynamic confidence is ', dynamic_confidence)
                cur_hat_ws.append(hat_w_2)
                flag = compare_model_signature(di_s=nn_shape, gamma_ps=gamma_p,
                                               extratced_ws=cur_hat_ws,
                                               l1_error=l1_error, confidence=dynamic_confidence * 0.95)
                if flag == 0:
                    # print('filtered by filter 3')
                    continue

                # increase the counter of the model candidate
                model_candidate_num += 1

                # step 5: recover biases
                index_set = [v for v in group_index]
                index_set.append(i)
                cur_hat_ws = [hat_w_1, hat_w_2]

                # get the biases using extracted weights, i.e., w^1, w^2
                bias_p = []
                for j in range(len(group_index)):
                    cur_i = group_index[j]
                    tp = np.matmul(hat_w_1[j], valid_boundary_points[cur_i])
                    cur_bias = -1 * hat_w_2[0][j] * np.sum(tp)
                    bias_p.append(cur_bias)
                # consider the boundary point that makes all the neurons active
                h = valid_boundary_points[i]
                h = np.matmul(hat_w_1, h)
                h = np.matmul(hat_w_2, h)
                h = np.sum(h)
                bias_p.append(-1.0 * h)

                # solve the system of linear equations
                hat_bs = recover_bs_via_soe(di_s=nn_shape, B_P=bias_p,
                                            group_index=index_set, extracted_ws=cur_hat_ws)

                print('we have found a solution')
                print('the model activation patterns of selected points are ', selected_maps)
                print('selected w_2 sign is ', w_2_sign)
                print('recovered w_1 is ', hat_w_1)
                print('recovered w_2 is ', hat_w_2)
                print('recovered bs are ', hat_bs)
                print('start checking the prediction matching ratio')
                prediction_matching_ratio = utils.check_prediction_matching_ratio(t_ws=weights,
                                                                                  t_bs=biases,
                                                                                  h_ws=cur_hat_ws,
                                                                                  h_bs=hat_bs)
                print('prediction matching ratio is ', prediction_matching_ratio)
                print('')

    print('{} models are checked, and {} extracted models are final candidates'.format(model_num,
                                                                                       model_candidate_num))


if __name__ == '__main__':
    precision = 10**(-10)
    l1_error = 10**(-3)

    nn_shape = [32, 3, 1]
    nn_path = './models/c1_1_deep_nn_32_3_1.npz'
    extract_1_deep_nn(model_path=nn_path, nn_shape=nn_shape, precision=precision, l1_error=l1_error)