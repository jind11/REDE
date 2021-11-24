from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import numpy as np
from oodMetrics import oodMetrics
import sys
from collections import defaultdict
import time

estimator_type = sys.argv[1]

def compute_kernel_bias(vecs, out_dim=None):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
#     vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    # return None, None
    # return W, -mu
    if out_dim is None:
        return W, -mu
    else:
        return W[:, :out_dim], -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

# OneclassSVM
# Gaussian density estimation with full covariance.
feats_tr = np.load(
    'sentence_embeddings/kmdm_train_distilbert-base-nli-stsb-mean-tokens_kmdm_mlm_epoch40_mean_embeddings.npy')
feats_tr_ood = np.load(
    'sentence_embeddings/kmdm_OODtrain_distilbert-base-nli-stsb-mean-tokens_kmdm_mlm_epoch40_mean_embeddings.npy')
feats_val = np.load(
    'sentence_embeddings/kmdm_val_distilbert-base-nli-stsb-mean-tokens_kmdm_mlm_epoch40_mean_embeddings.npy')
feats_test = np.load(
    'sentence_embeddings/kmdm_test_distilbert-base-nli-stsb-mean-tokens_kmdm_mlm_epoch40_mean_embeddings.npy')
feats_tripadvisor = np.load(
    'sentence_embeddings/kmdm_test_tripadvisor_distilbert-base-nli-stsb-mean-tokens_kmdm_mlm_epoch40_mean_embeddings.npy')
feats_subjective_questions = np.load(
    'sentence_embeddings/kmdm_subjective_questions_distilbert-base-nli-stsb-mean-tokens_kmdm_mlm_epoch40_mean_embeddings.npy')

# read data
lines = [line.strip().split('\t') for line in open('data/kmdm/eval.tsv').readlines()]
data = [(0 if line[0] == 'inDomain' else 1, line[-1]) for line in lines]
labels_val = np.array([entry[0] for entry in data])

lines = [line.strip().split('\t') for line in open('data/kmdm/test.tsv').readlines()]
data = [(0 if line[0] == 'inDomain' else 1, line[-1]) for line in lines]
labels = np.array([entry[0] for entry in data])

lines = [line.strip().split('\t') for line in open('data/kmdm/test_tripadvisor.tsv').readlines()]
data = [(0 if line[0] == 'inDomain' else 1, line[-1]) for line in lines]
labels_tripadvisor = np.array([entry[0] for entry in data])

lines = [line.strip().split('\t') for line in open('data/kmdm/subjective-questions.tsv').readlines()]
data = [(0 if line[0] == 'inDomain' else 1, line[-1]) for line in lines]
labels_subjective_questions = np.array([entry[0] for entry in data])

# matrix whitening
zero_ood = False
max_iter = 300
res_val = {'P': defaultdict(list), 'R': defaultdict(list), 'F1': defaultdict(list)}
res_test = {'P': defaultdict(list), 'R': defaultdict(list), 'F1': defaultdict(list)}
res_contrast = {'P': defaultdict(list), 'R': defaultdict(list), 'F1': defaultdict(list)}
res_subjective_questions = {'P': defaultdict(list), 'R': defaultdict(list), 'F1': defaultdict(list)}

for ood_num in [-1]:
    print(f'ood num: {ood_num}')
    for out_dim in [feats_tr.shape[-1]-1]:
        print('out_dim: {}'.format(out_dim))
        if not zero_ood:
            if ood_num < 0:
                kernel, bias = compute_kernel_bias(feats_tr_ood, out_dim=out_dim)
            elif ood_num == 0:
                kernel, bias = compute_kernel_bias(feats_tr, out_dim=out_dim)
            else:
                kernel, bias = compute_kernel_bias(feats_tr_ood[np.random.choice(feats_tr_ood.shape[0],
                                                                                 size=ood_num, replace=False), :],
                                                   out_dim=out_dim)
            feats_tr_ = transform_and_normalize(feats_tr, kernel, bias)
            feats_val_ = transform_and_normalize(feats_val, kernel, bias)
            feats_ = transform_and_normalize(feats_test, kernel, bias)
            feats_norm = normalize(feats_test)
            feats_tripadvisor_ = transform_and_normalize(feats_tripadvisor, kernel, bias)
            feats_subjective_questions_ = transform_and_normalize(feats_subjective_questions, kernel, bias)
        else:
            feats_tr_ = normalize(feats_tr)
            feats_val_ = normalize(feats_val)
            feats_ = normalize(feats_test)
            feats_tripadvisor_ = normalize(feats_tripadvisor)
            feats_subjective_questions_ = normalize(feats_subjective_questions)

        for n_components in range(1, 2, 1):
            print('n_components: {}'.format(n_components))
            if estimator_type == 'ocsvm':
                scaler = StandardScaler()
                estimator = OneClassSVM(kernel='rbf', gamma='auto', max_iter=max_iter)
            elif estimator_type.startswith('kde'):
                kernel = estimator_type.split('-')[-1]
                if kernel == 'exponential':
                    bandwidth = 0.2
                elif kernel == 'gaussian':
                    bandwidth = 0.4
                estimator = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            elif estimator_type == 'gmm':
                estimator = GMM(n_components=n_components, init_params='kmeans', covariance_type='full')
            elif estimator_type == 'bgm':
                estimator = BGM(weight_concentration_prior_type="dirichlet_process",
                n_components=n_components, reg_covar=0, init_params='kmeans',
                max_iter=max_iter, mean_precision_prior=.5, random_state=2)
            else:
                raise NotImplementedError
            estimator.fit(feats_tr_)

            print('evaluting val set')
            scores_val = -estimator.score_samples(feats_val_)
            oodMetricsObj = oodMetrics()
            pr_thres_best_f1 = oodMetricsObj.compute_all_metrics(labels_val, scores_val, choose_thresholds=True)
            oodMetricsObj.pretty_print_metrics()
            print(pr_thres_best_f1, '\n')
            P, R, F1 = oodMetricsObj.metrics['P Scores OOD'], oodMetricsObj.metrics['R Scores OOD'], oodMetricsObj.metrics['F1 Scores OOD']
            res_val['P'][out_dim].append(P)
            res_val['R'][out_dim].append(R)
            res_val['F1'][out_dim].append(F1)

            print('evaluting original test set')
            start_time = time.time()
            scores = -estimator.score_samples(feats_)
            end_time = time.time()
            print(f"---evaluating test set for {end_time - start_time} seconds ---")
            oodMetricsObj = oodMetrics()
            pr_thres_best_f1 = oodMetricsObj.compute_all_metrics(labels, scores, choose_thresholds=True,
                                                                 fixed_threshold=pr_thres_best_f1)
            oodMetricsObj.pretty_print_metrics()
            print(pr_thres_best_f1, '\n')
            P, R, F1 = oodMetricsObj.metrics['P Scores OOD'], oodMetricsObj.metrics['R Scores OOD'], oodMetricsObj.metrics[
                'F1 Scores OOD']
            res_test['P'][out_dim].append(P)
            res_test['R'][out_dim].append(R)
            res_test['F1'][out_dim].append(F1)

            print('evaluting tripadvisor test set')
            start_time = time.time()
            scores_tripadvisor = -estimator.score_samples(feats_tripadvisor_)
            end_time = time.time()
            print(f"---evaluating test set for {end_time - start_time} seconds ---")
            oodMetricsObj = oodMetrics()
            pr_thres_best_f1 = oodMetricsObj.compute_all_metrics(labels_tripadvisor, scores_tripadvisor,
                                                                 choose_thresholds=True, fixed_threshold=pr_thres_best_f1)
            oodMetricsObj.pretty_print_metrics()
            print(pr_thres_best_f1, '\n')
            P, R, F1 = oodMetricsObj.metrics['P Scores OOD'], oodMetricsObj.metrics['R Scores OOD'], oodMetricsObj.metrics[
                'F1 Scores OOD']
            res_contrast['P'][out_dim].append(P)
            res_contrast['R'][out_dim].append(R)
            res_contrast['F1'][out_dim].append(F1)

            print('evaluting subjective questions set')
            start_time = time.time()
            scores_subjective_questions = -estimator.score_samples(feats_subjective_questions_)
            end_time = time.time()
            print(f"---evaluating test set for {end_time - start_time} seconds ---")
            oodMetricsObj = oodMetrics()
            pr_thres_best_f1 = oodMetricsObj.compute_all_metrics(labels_subjective_questions, scores_subjective_questions,
                                                                 choose_thresholds=True,
                                                                 fixed_threshold=pr_thres_best_f1)
            oodMetricsObj.pretty_print_metrics()
            print(pr_thres_best_f1, '\n')
            P, R, F1 = oodMetricsObj.metrics['P Scores OOD'], oodMetricsObj.metrics['R Scores OOD'], \
                       oodMetricsObj.metrics[
                           'F1 Scores OOD']
            res_subjective_questions['P'][out_dim].append(P)
            res_subjective_questions['R'][out_dim].append(R)
            res_subjective_questions['F1'][out_dim].append(F1)

print('Val results: ', res_val)
print('Test results: ', res_test)
print('Contrast results: ', res_contrast)
print('Subjective questions results: ', res_subjective_questions)