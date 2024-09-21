# NN_cryptanalytic_extraction

Hard-Label Cryptanalytic Extraction of Neural Network Models.
https://arxiv.org/abs/2409.11646v1
https://eprint.iacr.org/2024/1403
Yi Chen, Xiaoyang Dong, Jian Guo, Yantian Shen, Anyu Wang, Xiaoyun Wang

requirements: python 3.7.0, numpy 1.21.5

check_solution_svd.py: 
compute $(\varepsilon, 0)$-functional equivalence
via error bounds propagation.

verify_1_deep_nn_attack.py
(verify_2_deep_nn_attack.py):
verify the experiment results in Tables 1 and 2.

./models/: 
victim models (including untrained and trained)
and extracted models.

./attack_records/: 
records of model extraction attacks.
