import sys

import numpy as np
from sklearn.model_selection import train_test_split
from utils import (
    my_seed,
    Dataprocessor,
    dataset_report,
    generate_meta_label,
    final_model_report
)
from model_wrapper import ModelWrapper, ModelAnalyzer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# alpha_list = ['buysellratio', 'alpha2', 'alpha9', 'alpha25', 'alpha32', 'alpha46', 'alpha104', 'rolling_1_alpha2',
#               'rolling_5_alpha2', 'rolling_10_alpha2', 'rolling_1_alpha9', 'rolling_5_alpha9', 'rolling_1_alpha25',
#               'rolling_5_alpha25', 'rolling_1_alpha32', 'rolling_5_alpha32', 'rolling_1_alpha46', 'rolling_5_alpha46',
#               'upsidevolumeratio', 'alpha1', 'alpha95', 'alpha101', 'alpha102', 'alpha103', 'alpha105', 'alpha106',
#               'alpha107', 'rolling_10_label', 'rolling_1_alpha95', 'rolling_5_alpha95', 'rolling_1_alpha101',
#               'rolling_5_alpha101', 'rolling_10_alpha101','rolling_1_alpha106','rolling_5_alpha106','rolling_1_alpha107']
alpha_list = ['buysellratio', 'upsidevolumeratio',  'alpha1', 'alpha2', 'alpha9', 'alpha25', 'alpha32', 'alpha46', 'alpha95', 'alpha101', 'alpha102', 'alpha103', 'alpha104', 'alpha105', 'alpha106', 'alpha107', 'rolling_1_buysellratio', 'rolling_1_upsidevolumeratio', 'rolling_1_alpha1', 'rolling_5_alpha1', 'rolling_1_alpha2', 'rolling_5_alpha2', 'rolling_10_alpha2', 'rolling_1_alpha9', 'rolling_5_alpha9', 'rolling_1_alpha25', 'rolling_5_alpha25', 'rolling_1_alpha32', 'rolling_5_alpha32', 'rolling_1_alpha46', 'rolling_5_alpha46', 'rolling_1_alpha95', 'rolling_5_alpha95', 'rolling_1_alpha101', 'rolling_5_alpha101', 'rolling_10_alpha101', 'rolling_1_alpha103', 'rolling_1_alpha106', 'rolling_5_alpha106', 'rolling_10_alpha106', 'rolling_1_alpha107', 'rolling_5_alpha107', 'rolling_10_alpha107']

#alpha_list = ['rolling_10_label']

train_processor = Dataprocessor(alpha_list)
feature, label, features = train_processor.load_dataset("data_rolling.pkl")
# features = features[:len(features)//10]
# X = np.where(np.isinf(X), 10, X)
# first = X[:, 0]
# X = X[:, 1:]
# del features[0]

# print(dataset_report(X, y, features))

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.1, random_state=42)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
all_y_true = []
all_y_pred = []

wrapper = ModelWrapper("RF")
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    wrapper.train(X_train_fold, y_train_fold)
    y_pred = wrapper.model.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, y_pred)
    all_y_true.extend(y_val_fold)
    all_y_pred.extend(y_pred)

overall_report = classification_report(all_y_true, all_y_pred, digits=2)

print("\n================ 综合 K 折交叉验证报告 ================\n")
print(overall_report)
analyser = ModelAnalyzer("RF", wrapper.model, X_train, y_train, X_test, y_test)
# train_pred, test_pred, _, _ = analyser.generate_predictions(prediction_mode='class')
analyser.print_report(prediction_mode='class')
# print(train_pred)
# print(test_pred)
# print(np.concatenate((train_pred, test_pred)))
# print(processor.load_prediction(np.concatenate((train_pred, test_pred))))
# plot_by_asset(processor.dataset)
# processor.save()
# meta_label_train = generate_meta_label(train_pred, train_label)
# meta_label_test = generate_meta_label(test_pred, test_label)
#
# train_meta_feature = np.concatenate((train_feature, train_pred.reshape(-1, 1)), axis=1)
# test_meta_feature = np.concatenate((test_feature, test_pred.reshape(-1, 1)), axis=1)
#
# # meta_train_feature = np.concatenate((meta_train_feature.reshape(-1, 1), train_pred.reshape(-1, 1)), axis=1)
#
# meta_wrapper = ModelWrapper("RF", train_feature=train_meta_feature, train_label=meta_label_train)
#
# hyper_params = {
#     'n_estimators': 200,
#     'min_samples_split': 2,
#     'min_samples_leaf': 2,
#     'max_depth': 10,
#     'max_features': 'sqrt',
#     'max_samples': 0.7,
#     'random_state': 42,
#     'class_weight': 'balanced_subsample',
#     'bootstrap': True,
#     'n_jobs': 16
# }
# meta_wrapper.set_model_params(hyper_params)
# meta_wrapper.train()
# meta_analyser = ModelAnalyzer(meta_wrapper.model, train_meta_feature, meta_label_train, test_meta_feature,
#                               meta_label_test)
# meta_train_pred, meta_test_pred, _, _ = meta_analyser.generate_predictions(prediction_mode='class')
# meta_analyser.print_report()

# final_model_report(train_pred,meta_train_pred,train_label)
#
# final_model_report(test_pred,meta_test_pred,test_label)

# 直接打印整个报告字典

# 或者按项目逐项打印，更加清晰


# combined_train_feature = np.concatenate((train_feature, meta_train_pred.reshape(-1, 1)), axis=1)
# combine_test_feature = np.concatenate((test_feature, meta_test_pred.reshape(-1, 1)), axis=1)
#
# final_wrapper = ModelWrapper("RF", train_feature=combined_train_feature, train_label=train_label)
# final_wrapper.train()
# final_analyser = ModelAnalyzer(final_wrapper.model, combined_train_feature, train_label, combine_test_feature,
#                                test_label)
#
# final_analyser.print_report()

# train_result, test_result, train_uncertainty, test_uncertainty = wrapper.generate_predictions(prediction_mode='class',
#                                                                                               threshold=None)
# wrapper.save()
import joblib

# joblib.dump(wrapper.model, r"models\MLP-20250213-201721.pkl")
# model = wrapper.model
#
# print_model_report(
#     wrapper.model,
#     X_train=X_train,
#     y_train=y_train,
#     X_test=X_test,
#     y_test=y_test,
#     use_arg_to_predict=False,  # MLP模型设为True，其余模型设为False
#     features=features,
#     threshold=0.5
# )
# wrapper.save()
