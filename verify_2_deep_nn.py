'''
2-deep NN extraction attack
'''

import numpy as np
import random
from itertools import combinations
import utils


def expected_nn_parameters(nn_shape, ws):
    print('when the recovered model is correct: ')
    print('expected w_1 is: ')
    for i in range(nn_shape[1]):
        print(ws[0][i] / abs(ws[0][i][0]))

    print('expected w_2 is: ')
    tp = np.array([ws[0][i][0] for i in range(nn_shape[1])]).reshape(-1, 1)
    tp = np.matmul(ws[1], tp)  # array shape is [d_2, d_1] * [d_1, 1] --> [d_2, 1]
    expected_w_2 = []
    for i in range(nn_shape[2]):
        tp_w_2_i = []
        for j in range(nn_shape[1]):
            tp_w_2_i.append(ws[1][i][j] * abs(ws[0][j][0] / tp[i][0]))
        expected_w_2.append(tp_w_2_i)
    print(expected_w_2)

    print('expected w_3 is: ')
    tp_new = np.matmul(ws[2], tp)
    expected_w_3 = []
    for i in range(nn_shape[3]):
        tp_w_3_i = []
        for j in range(nn_shape[2]):
            tp_w_3_i.append(ws[2][i][j] * abs(tp[j][0] / tp_new[i][0]))
        expected_w_3.append(tp_w_3_i)
    print(expected_w_3)

    # print('when we regard k-deep nn as a zero-deep nn: ')
    # print('expected (gamma_p, B_p)s are: ')
    # for i in range(nn_shape[1]):
    #     print(ws[0][i] * ws[1][0][i] / abs(ws[0][i][0] * ws[1][0][i]))


# recover the affine transofrmation of the k-deep NN
# ws, bs are the true nn parameters used to collect decision boundary points
def recover_gamma_b(x, ws, bs, ms=10 ** (-3), precision=10 ** (-8)):
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
    hat_ws_bs = np.zeros((1, d_0 + 1), dtype=np.float64)
    hat_ws_bs[0][0] = ws_signs[0] * 1.0

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
            hat_ws_bs[0][j] = -1 * ws_signs[0] * ms * hat_ws_bs[0][0] / suitable_ms

    # recover the bias
    for i in range(d_0):
        hat_ws_bs[0][d_0] -= hat_ws_bs[0][i] * x[i][0]

    return hat_ws_bs


# recover the weight vector of the j-th neuron in layer i, by solving a system of linear equations
def recover_ws_via_soe(di_s, extracted_ws, target_ws, layer_no, selected_indexs):
    '''
    :param di_s: the list consisting of the dimension in each layer
    :param extracted_ws: extracted weights in layers 1,...,i-1, the array shape is [i-1, d_j, d_{j-1}]
    :param target_ws: the weight vector (\gamma_P) of the affine transformation, shape is [1, d_0]
    :param layer_no: the number of the attacked layer, i.e., i
    :return: extracted weight vector of the target neuron in layer i
    '''
    assert layer_no - 1 == len(extracted_ws)
    d_0 = di_s[0]
    d_i_minus_1 = di_s[layer_no - 1]
    assert d_0 >= d_i_minus_1

    # build the coefficient array
    h = extracted_ws[layer_no - 2]
    for i in range(layer_no - 3, -1, -1):
        h = np.matmul(h, extracted_ws[i])
    h = h.T  # array shape is [d_0, d_i_minus_1]

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
def recover_bs_via_soe(di_s, B_P, group_index, extracted_ws, mp_2):
    '''
    :param di_s: the list consisting of the dimension in each layer
    :param B_P: recovered n biases B_p of f(x) = \gamma_p x + B_p, the array shape is (n, 1)
    :param group_index: the list of selected decision boundary points,
                        [[i_{1,1},...,i_{1,d_1}], [i_{2,1},...,i_{2,d_2}],...,[i_{k+1, 1}]]
    :param extracted_ws: extracted weights, the array shape is [k+1, d_i, d_{i-1}]
    :return:
    '''
    layer_num = len(di_s) - 1
    assert len(extracted_ws) == layer_num == 3

    # neuron num
    n = np.sum(np.array(di_s)[1:layer_num])
    # build the coefficients of [b_1^1,...,b_{d_1}^1, b_1^2,...,b_{d_2}^2, ...  , b_1^{k+1}]
    h = np.zeros((n + 1, n + 1), dtype=np.float64)

    # consider the n decision boundary points,
    # not including the one that makes all the neurons active
    cnt = 0

    # d1 points
    assert di_s[1] == di_s[2] == 2
    # adjust according to mp_2 ([0, 1] or [1, 0] or [0, 0] or [1, 1])
    h[cnt][0] = extracted_ws[2][0][mp_2[0]] * extracted_ws[1][mp_2[0]][0]
    h[cnt][1] = 0
    if mp_2[0] == 0:
        h[cnt][2] = extracted_ws[2][0][mp_2[0]]
        h[cnt][3] = 0
    else:
        h[cnt][2] = 0
        h[cnt][3] = extracted_ws[2][0][mp_2[0]]
    h[cnt][4] = 1
    cnt += 1
    # ------------------------------------------
    h[cnt][0] = 0
    h[cnt][1] = extracted_ws[2][0][mp_2[1]] * extracted_ws[1][mp_2[1]][1]
    if mp_2[1] == 0:
        h[cnt][2] = extracted_ws[2][0][0]
        h[cnt][3] = 0
    else:
        h[cnt][2] = 0
        h[cnt][3] = extracted_ws[2][0][0]
    h[cnt][4] = 1
    cnt += 1

    # d2 points
    for i in range(di_s[2]):
        for j in range(di_s[1]):
            h[cnt][j] = extracted_ws[2][0][i] * extracted_ws[1][i][j]

        h[cnt][di_s[1] + i] = extracted_ws[2][0][i]
        h[cnt][n] = 1
        cnt += 1

    # consider the decision boundary point that makes all the neurons active
    assert cnt == n
    for i in range(di_s[1]):
        tp = np.array([extracted_ws[2][0][j] * extracted_ws[1][j][i] for j in range(di_s[2])])
        h[cnt][i] = np.sum(tp)
    for i in range(di_s[2]):
        h[cnt][di_s[1] + i] = extracted_ws[2][0][i]
    h[cnt][n] = 1

    # build the value array
    y = B_P

    # solving the system of linear equations
    soln, *rest = np.linalg.lstsq(h, y, 1e-6)

    # reshape soln, [k+1, d_i, 1]
    bs = []
    cnt = 0
    for i in range(layer_num):
        b_i = []
        di = di_s[i + 1]
        for j in range(di):
            b_i.append(soln[cnt])
            cnt += 1
        b_i = np.array(b_i, dtype=np.float64).reshape((-1, 1))
        bs.append(b_i)
        # bs.append(np.array(b_i, dtype=np.float64))

    return bs


def compare_model_signature(di_s, gamma_ps, extratced_ws, l1_error=10 ** (-3), confidence=0.6):
    '''
    :param gamma_ps: the true weight vectors of affine transformations, the array shape is [, 1, d_0]
    :param extratced_ws: extracted weights in each layer, the array shape is [, d_i, d_{i-1}]
    :return: whether the extracted weights and biases are the true one
    '''
    hidden_layer_num = len(di_s) - 2
    n = np.sum(np.array(di_s)[1:hidden_layer_num + 1])

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
        ans = [res[i - 1] for i in range(hidden_layer_num, 0, -1)]
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
    total_num = 2 ** n
    for p in range(total_num):
        DMs = [np.zeros((di_s[i + 1], di_s[i + 1]), dtype=np.float64) for i in range(hidden_layer_num)]
        ps = convert_scalar_to_arrays(p)
        if check_validity(ps) == 0:
            continue

        # compute the diagonal matrix
        for i in range(hidden_layer_num):
            di = di_s[i + 1]
            for j in range(di):
                if ps[i][j] == 1:
                    DMs[i][j][j] = 1

        # compute the gamma_p
        h = extratced_ws[hidden_layer_num]
        for i in range(hidden_layer_num, 0, -1):
            h = np.matmul(h, DMs[i - 1])
            h = np.matmul(h, extratced_ws[i - 1])
        assert h.shape[0] == 1 and h.shape[1] == di_s[0]

        # ignore the gamma_p with h[0][0] = 0 temporarily
        if h[0][0] != 0:
            extracted_gamma_ps.append(h / abs(h[0][0]))
    ps_num = len(extracted_gamma_ps)

    # chech whether the true gamma_ps is a subset of the collected gamma_ps
    gamma_ps_num = len(gamma_ps)
    cnt = 0
    for i in range(gamma_ps_num):  # the true model signature
        flag = 0
        for j in range(ps_num):  # the extracted model signature
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


def get_dynamic_confidence(complete_gamma_p, unique_indexs, l1_error=10 ** (-3)):
    gamma_num = len(complete_gamma_p)
    unique_index_num = len(unique_indexs)
    cnt = np.zeros(unique_index_num, dtype=np.uint32)
    for i in range(unique_index_num):
        index = unique_indexs[i]
        for j in range(gamma_num):
            if j in unique_indexs:
                cnt[i] += 1
                continue
            if np.max(np.abs(complete_gamma_p[j] - complete_gamma_p[index])) < l1_error:
                cnt[i] += 1
    suitable_confidence = 1 - np.sum(cnt < 2) / unique_index_num
    return suitable_confidence


def identify_final_model_candidate(surviving_models=None):
    nn_num = len(surviving_models)
    if nn_num == 0:
        print('no surviving nn models')
        return None
    else:
        PMRs = [surviving_models[i][0] for i in range(nn_num)]
        index = PMRs.index(max(PMRs))
        pmr, ws_extracted, bs_extracted = surviving_models[index][0], \
                                          surviving_models[index][1], \
                                          surviving_models[index][2]
        return pmr, ws_extracted, bs_extracted


def extract_2_deep_nn(model_path=None, nn_shape=None, precision=10 ** (-8), l1_error=10 ** (-3)):
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
    n = 8 * (2 ** neuron_num)

    # start the extraction attack
    # steps 1: collect decision boundary points
    print('start step 1: collect decision boundary points')
    tmp_boundary_points = []
    tmp_maps = []
    cnt = 0
    while cnt < n:
        # x is a decision boundary point
        x = utils.find_one_decision_boundary_point(ws=weights, bs=biases, layer_num=2,
                                                   precision=precision)
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

    # step 2: recover (\gamma_p, B_p)
    print('start step 2: recover (\gamma_P, B_P)')
    valid_boundary_points = []
    gamma_b = []
    maps = []
    for i in range(tmp_boundary_point_num):
        x = tmp_boundary_points[i]
        # hat_ws, hat_ws shape: (1, d_0)
        hat_ws_bs = recover_gamma_b(x=x, ws=weights, bs=biases,
                                    ms=10 ** (-3), precision=precision)
        if hat_ws_bs is not None:
            valid_boundary_points.append(x)
            gamma_b.append(hat_ws_bs)
            maps.append(tmp_maps[i])
    gamma_b = np.array(gamma_b)

    # filter decision boundary points with duplicate model activation patterns
    # whether two MAPs are the same are verified by their parameters \gamma_p of the affine transformation
    if len(gamma_b) != 0:
        unique_indexs = utils.filter_duplicate_vectors(arr=gamma_b, l1_error=l1_error)
        print('{} different MAPs occur'.format(len(unique_indexs)))

        print('corresponding (\gamma_P, B_P) are: ')
        for i in unique_indexs:
            print('boundary point: {}, (\gamma_p, B_P) is {}'.format(i, gamma_b[i]))

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
    survive_models = []

    model_num = 0
    model_candidate_num = 0
    # select d_1 + 1 decision boundary points each time
    # first select d_1 decision boundary points that only makes the i-th neuron active in layer 1
    for group_index_1 in combinations(unique_indexs, nn_shape[1]):
        # select d_2 decision boundary points that only makes the j-th neuron active in layer 2
        remaining_indexs = [i for i in unique_indexs if i not in group_index_1]
        for group_index_2 in combinations(remaining_indexs, nn_shape[2]):
            # select 1 decision boundary point that makes all the neurons active
            for index_3 in unique_indexs:
                if index_3 in group_index_1 or index_3 in group_index_2:
                    continue
                for ws_signs in [[1, 1], [1, -1], [-1, 1], [-1, -1]]:
                    # increase the counter of tested model
                    model_num += 1
                    selected_maps = [maps[v] for v in group_index_1]
                    for v in group_index_2:
                        selected_maps.append(maps[v])
                    selected_maps.append(maps[index_3])

                    # to compute functionally equivalence, choose the correct one
                    valid_maps = [2**k for k in range(nn_shape[1])]
                    flag = 1
                    for v in group_index_1:
                        mp_in_layer_1 = maps[v]
                        if mp_in_layer_1[0] not in valid_maps:
                            flag = flag * 0
                            break
                    if flag == 0:
                        continue

                    # step 3: recover weights w_1 in layer 1
                    # print('start step 3')
                    hat_w_1 = [ws_signs[0] * gamma_b[cur_i][0][:nn_shape[0]] for cur_i in group_index_1]
                    hat_w_1 = np.array(hat_w_1, dtype=np.float64)

                    # step 4-1: recover weights w_2 in layer 2
                    # print('start step 4')
                    # cur_hat_ws = np.array([hat_w_1], dtype=np.float64)
                    cur_hat_ws = [hat_w_1]
                    hat_w_2 = []
                    for index_2 in group_index_2:
                        hat_w_2_index_2 = recover_ws_via_soe(di_s=nn_shape,
                                                             extracted_ws=cur_hat_ws,
                                                             target_ws=gamma_b[index_2][:nn_shape[0]] * ws_signs[1],
                                                             layer_no=2,
                                                             selected_indexs=[j for j in range(nn_shape[1])])
                        hat_w_2.append(hat_w_2_index_2[0])
                    hat_w_1 = np.array(hat_w_1, dtype=np.float64)

                    # step 4-2: recover weights w_3 in layer 3
                    # cur_hat_ws = np.array([hat_w_1, hat_w_2], dtype=np.float64)
                    cur_hat_ws = [hat_w_1, hat_w_2]
                    hat_w_3 = recover_ws_via_soe(di_s=nn_shape,
                                                 extracted_ws=cur_hat_ws,
                                                 target_ws=gamma_b[index_3][:nn_shape[0]],
                                                 layer_no=3,
                                                 selected_indexs=[j for j in range(nn_shape[2])])

                    # filter 2: the ws_signs of recovered NN should be the same as the expectation
                    if np.any(hat_w_3 * ws_signs[1] < 0):
                        # print('filtered by filter 2')
                        continue

                    # hat_w_3_2 = np.matmul(hat_w_3, hat_w_2)
                    # if np.any(hat_w_3_2 * ws_signs[0] < 0):
                    #     # print('filtered by filter 2')
                    #     continue

                    # filter 3: check whether the model signature of extracted nn is equivalent to the true one
                    dynamic_confidence = get_dynamic_confidence(complete_gamma_p=gamma_b,
                                                                unique_indexs=unique_indexs,
                                                                l1_error=l1_error)
                    # print('dynamic confidence is ', dynamic_confidence)
                    # print('gamma_b shape is ', np.shape(gamma_b))
                    cur_hat_ws = [hat_w_1, hat_w_2, hat_w_3]
                    flag = compare_model_signature(di_s=nn_shape,
                                                   gamma_ps=gamma_b[:, :, :nn_shape[0]],
                                                   extratced_ws=cur_hat_ws,
                                                   l1_error=l1_error, confidence=dynamic_confidence * 0.95)
                    if flag == 0:
                        # print('filtered by filter 3')
                        continue

                    # increase the counter of the model candidate
                    model_candidate_num += 1

                    # step 5: recover biases
                    index_set = [v for v in group_index_1]
                    for tp_i in group_index_2:
                        index_set.append(tp_i)
                    index_set.append(index_3)

                    assert nn_shape[2] == 2
                    for mp_2 in [[0, 1], [1, 0], [0, 0], [1, 1]]:
                        # get the biases using extracted weights, i.e., w^1, w^2
                        bias_p = []
                        # for the d_1 points corresponding to layer 1
                        # for j in range(nn_shape[1]):
                        #     cur_i = group_index_1[j]
                        #     tp = np.matmul(hat_w_3, hat_w_2)
                        #     tp = tp[0][j] * np.matmul(hat_w_1[j], valid_boundary_points[cur_i])
                        #     cur_bias = -1.0 * np.sum(tp)
                        #     bias_p.append(cur_bias)

                        # adjusted according to mp_2
                        cur_i_0 = group_index_1[0]
                        tp = hat_w_3[0][mp_2[0]] * hat_w_2[mp_2[0]][0]
                        tp = tp * np.matmul(hat_w_1[0], valid_boundary_points[cur_i_0])
                        cur_bias = -1.0 * np.sum(tp)
                        bias_p.append(cur_bias)

                        cur_i_1 = group_index_1[1]
                        tp = hat_w_3[0][mp_2[1]] * hat_w_2[mp_2[1]][1]
                        tp = tp * np.matmul(hat_w_1[1], valid_boundary_points[cur_i_1])
                        cur_bias = -1.0 * np.sum(tp)
                        bias_p.append(cur_bias)

                        # for the d_2 points corresponding to layer 2
                        for j in range(nn_shape[2]):
                            cur_i = group_index_2[j]
                            tp = np.matmul(hat_w_3[0][j] * hat_w_2[j], hat_w_1)
                            tp = np.matmul(tp, valid_boundary_points[cur_i])
                            cur_bias = -1.0 * np.sum(tp)
                            bias_p.append(cur_bias)

                        # for the point that makes all the neurons active
                        cur_i = index_3
                        tp = np.matmul(hat_w_3, hat_w_2)
                        tp = np.matmul(tp, hat_w_1)
                        tp = np.matmul(tp, valid_boundary_points[cur_i])
                        cur_bias = -1.0 * np.sum(tp)
                        bias_p.append(cur_bias)

                        # solve the system of linear equations
                        hat_bs = recover_bs_via_soe(di_s=nn_shape, B_P=bias_p,
                                                    group_index=index_set,
                                                    extracted_ws=cur_hat_ws, mp_2=mp_2)

                        print('we have found a solution')
                        print('the model activation patterns of selected points are ', selected_maps)
                        print('selected ws_signs is ', ws_signs)
                        print('recovered w_2 is ', hat_w_2)
                        print('start checking the prediction matching ratio')
                        prediction_matching_ratio = utils.check_prediction_matching_ratio(t_ws=weights,
                                                                                          t_bs=biases,
                                                                                          h_ws=cur_hat_ws,
                                                                                          h_bs=hat_bs)
                        print('prediction matching ratio is ', prediction_matching_ratio)
                        print('')

                        # save the surviving model
                        survive_models.append([prediction_matching_ratio, cur_hat_ws, hat_bs])

    print('{} models are checked, and {} extracted models are final candidates'.format(model_num,
                                                                                       model_candidate_num))

    # identify and save the final candidate of extracted nn model
    pmr, ws_extracted, bs_extracted = identify_final_model_candidate(surviving_models=survive_models)

    # folder = './models/extracted_models/'
    # tmp_path = ''
    # for i in range(len(nn_shape)):
    #     tmp_path += '_' + str(nn_shape[i])
    # model_path = folder + '2_deep_nn' + tmp_path
    # np.savez(model_path, ws_extracted, bs_extracted)


if __name__ == '__main__':
    precision = 10**(-10)
    l1_error = 10**(-3)

    nn_shape = [8, 2, 2, 1]
    nn_path = './models/real_models/c1_2_deep_nn_8_2_2_1.npz'
    extract_2_deep_nn(model_path=nn_path, nn_shape=nn_shape, precision=precision, l1_error=l1_error)