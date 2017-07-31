# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from pymongo import MongoClient
from pymongo import MongoReplicaSetClient
from pymongo import ReadPreference
import traceback
import socket
from email.mime.text import MIMEText
import smtplib
import logging
import os
from sklearn import cluster, covariance, manifold, metrics
from sklearn import cross_validation
from sklearn.grid_search import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import datetime
# import xgboost




###### logger module ##########
# logger = logging.getLogger('logger_name')
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('[%(asctime)s %(levelname)s]: %(message)s')
# fh = logging.FileHandler(os.path.join(os.getcwd(), 'logger_name'))
# ch = logging.StreamHandler()
# fh.setFormatter(formatter)
# ch.setFormatter(formatter)
# logger.addHandler(fh)
# logger.addHandler(ch)

# ##### 连接 mysql 模块
# import MySQLdb
# with MySQLdb.connect(host='127.0.0.1', port=3306, user='root', passwd='lr187703', db='tieba_new_word_model') as conn:
#     cur = conn.cursor()
#     conn.set_character_set('utf8')
#     cur.execute('SET NAMES utf8;')
#     cur.execute('SET CHARACTER SET utf8;')
#     cur.execute('SET character_set_connection=utf8;')




def affinity_propagation_cluster(sim_matrix, variate_list, preference_value=1):
    """
    用affinity_propagation算法聚类并画图
    :param sim_matrix:
    :return:
    """

    grouping, labels = cluster.affinity_propagation(sim_matrix, preference=preference_value*np.median(sim_matrix))
    result_df = pd.DataFrame()
    result_df['variate'] = variate_list
    result_df['cluster'] = list(labels)
    result_df['variate'] = result_df['variate'].map(lambda i: "%s, " % i)
    result_df = result_df.groupby(['cluster']).sum()
    result_df['variate'] = result_df['variate'].map(lambda j: j[:-2])

    print "==="*10, u"聚类分组结果输出", "==="*10
    for cluster_index, values_data in result_df.iterrows():
        print('Cluster %i: %s' % ((cluster_index + 1), values_data.to_dict().get('variate')))
    return result_df


def cross_validate_model_param(clf, param, dataset, label):
    """
    将cross validation验证模型的结果存入数据库
    :param clf:  模型
    :param param: 参数
    :param dataset: x数据
    :param label: y数据
    :param k_fold:
    :param scoring:
    :param table: table name
    :return:
    """

    scores = cross_validation.cross_val_score(clf, X=dataset, y=label, scoring=param['scoring'], cv=param['cv'])
    param.update({"score": scores.mean(), 'updt': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    print "%s: %s" % (param['scoring'], scores.mean())
    with MongodbUtils(ip="localhost", port=37017, collection="model_result", table=param['table']) as col:
        col.insert(param)

def cross_validate_model_param_v2(clf, param, dataset, label):
    """
    将cross validation验证模型的结果存入数据库
    :param clf:  模型
    :param param: 参数
    :param dataset: x数据
    :param label: y数据
    :param k_fold:
    :param scoring:
    :param table: table name
    :return:
    """

    # scores = cross_validation.cross_val_score(clf, X=dataset, y=label, scoring=param['scoring'], cv=param['cv'])
    rst = []
    k_fold = cross_validation.KFold(n=dataset.shape[0], n_folds=param['cv'], shuffle=True)
    for train_idx, test_idx in k_fold:
        x_train = dataset[train_idx]
        y_train = label[train_idx]
        x_test = dataset[test_idx]
        y_test = label[test_idx]
        clf.fit(x_train, y_train)
        y_train_predict = clf.predict(x_train)
        y_test_predict = clf.predict(x_test)
        score_dict = {}
        print metrics.classification_report(y_test, y_test_predict)
        if "f1" in param['scoring']:
            f1_train_score = metrics.f1_score(y_train, y_train_predict)
            f1_test_score = metrics.f1_score(y_test, y_test_predict)

            score_dict.update({"f1_train_score": f1_train_score, "f1_test_score": f1_test_score})
        if "roc" in param['scoring']:
            roc_train_score = metrics.roc_auc_score(y_train, y_train_predict)
            roc_test_score = metrics.roc_auc_score(y_test, y_test_predict)
            score_dict.update({"roc_train_score": roc_train_score, "roc_test_score": roc_test_score})
        if "precision" in param['scoring']:
            precision_train_score = metrics.precision_score(y_train, y_train_predict)
            precision_test_score = metrics.precision_score(y_test, y_test_predict)
            score_dict.update({"precision_train_score": precision_train_score, "precision_test_score": precision_test_score})
        if "recall" in param['scoring']:
            recall_train_score = metrics.recall_score(y_train, y_train_predict)
            recall_test_score = metrics.recall_score(y_test, y_test_predict)
            score_dict.update({"recall_train_score": recall_train_score, "recall_test_score": recall_test_score})
        if "accuracy" in param['scoring']:
            recall_train_score = metrics.accuracy_score(y_train, y_train_predict)
            recall_test_score = metrics.accuracy_score(y_test, y_test_predict)
            score_dict.update({"accuracy_train_score": recall_train_score, "accuracy_test_score": recall_test_score})
        rst.append(score_dict)
    rst_df = pd.DataFrame(rst)
    mean_score = rst_df.mean()
    mean_score_dict = mean_score.to_dict()
    param.update(mean_score_dict)
    # param.update({'score_detailed': rst_df})
    param.update({'updt': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    print mean_score
    with MongodbUtils(ip="localhost", port=37017, collection="model_result", table=param['table']) as col:
        col.insert(param)


def grid_search_param_model(model_name, grid_search_param_dict, X, y, data_process_param_dict):
    param_grid = list(ParameterGrid(grid_search_param_dict))
    if model_name == "tree":
        for param in param_grid:
            tree_clf = DecisionTreeClassifier(max_depth=param.get("max_depth", None))
            param.update({'model': model_name})
            print param
            param.update(data_process_param_dict)
            cross_validate_model_param_v2(tree_clf, param, dataset=X, label= y)
    if model_name == "adaboosting":
        for param in param_grid:
            tree_clf = DecisionTreeClassifier(max_depth=param.get("max_depth", None))
            ada_boost_clf = AdaBoostClassifier(base_estimator=tree_clf, n_estimators=param.get("n_estimators", None), learning_rate=param.get("learning_rate", None))
            param.update({'model': model_name})
            print param
            param.update(data_process_param_dict)
            cross_validate_model_param_v2(ada_boost_clf, param, dataset=X, label= y)
    elif model_name == 'linearsvc':
        for param in param_grid:
            print "%s" % param
            param.update({'model': model_name})

            linsvc_clf = LinearSVC(C=param.get("C", None))
            if "class_weight" in param.keys():
                class_weight = param['class_weight']
                class_weight_key = class_weight.keys()
                class_weight_key = [unicode(key) for key in class_weight_key]
                class_weight_value = class_weight.values()
                class_weight = dict(zip(class_weight_key, class_weight_value))
                param.update({'class_weight': class_weight})
            param.update(data_process_param_dict)
            cross_validate_model_param_v2(linsvc_clf, param, dataset=X, label= y)
    elif model_name == "random_forest":
        for param in param_grid:
            print "%s" % param
            param.update({'model': model_name})
            rf_clf = RandomForestClassifier(n_estimators=param.get("n_estimators", None), max_depth=param.get("max_depth"))
            param.update(data_process_param_dict)
            cross_validate_model_param_v2(rf_clf, param, dataset=X, label=y)
    elif model_name == "svm":
        for param in param_grid:
            print "%s" % param
            param.update({'model': model_name})
            svm_clf = SVC(C=param.get("C"), gamma=param.get("gamma"))
            param.update(data_process_param_dict)
            cross_validate_model_param_v2(svm_clf, param, dataset=X, label=y)


def calculate_cos_sim_matrix(v_arr):
    """
    根据向量数组计算cos相似度矩阵
    :param v_arr:
    :return:
    """
    v_module = np.array([[np.sqrt(tieba_v.dot(tieba_v))] for tieba_v in v_arr])
    v_normal = np.divide(v_arr, v_module)
    forum_similarity = v_normal.dot(v_normal.T)
    return forum_similarity

