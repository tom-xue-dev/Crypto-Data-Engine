import sys
from datetime import datetime
import os
import pickle

import numpy as np

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

my_seed = 42
alpha_list = [
    "buysellratio",
    'alpha1',
    'alpha2',
    'alpha9',
    'alpha25',
    'alpha32',
    'alpha46',
    'alpha95',
    'alpha101',
    'alpha102',
    'alpha103',
    'alpha105',
    'alpha106',
    'alpha107'
]
pattern_list = [
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLMATHOLD",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]


def generate_meta_label(predict_label, true_label):
    meta_label = np.where(predict_label == true_label, 1, 0)
    return meta_label


class Dataprocessor:
    def __init__(self, select_feature=None):
        self.dataset = None
        if select_feature:
            self.select_features = select_feature
        else:
            self.select_features = None

    def load_dataset(self, filename, excludes=None):
        with open(filename, "rb") as file:
            data: pd.DataFrame = pickle.load(file)
            self.dataset = data
        if self.select_features:
            keys = self.select_features
        else:
            keys = list(data.columns)
            try:
                keys.remove("label")
            except ValueError:
                print('dataset error, lost of label column')
                sys.exit(0)

        if excludes:
            for exclude in excludes:
                try:
                    keys.remove(exclude)
                except ValueError:
                    print(f"notice column{exclude} not in dataset but you still want to remove it")

        data_arr = []
        for key in keys:
            if key in data.columns:
                feature = data[key].values  # 直接访问
                np_feature = np.asarray(feature).reshape(-1, 1)
                data_arr.append(np_feature)
            else:
                print(f"Notice: Column '{key}' not in dataset but you still want to choose it as a feature.")
        return np.concatenate(data_arr, axis=1), np.asarray(data["label"].values), keys

    def load_prediction(self, labels):
        self.dataset['signal'] = labels
        return self.dataset

    def save(self):
        with open('data_signal.pkl', 'wb') as f:
            pickle.dump(self.dataset, f)


def select_features(X, y, seed=42):
    # 为演示生成随机数据
    np.random.seed(seed)

    # 定义随机森林分类器（可以调整参数以适应你的数据）
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    # 定义 Boruta 特征选择器
    # n_estimators='auto' 会自动设置随机森林中树的数量
    boruta_selector = BorutaPy(
        estimator=rf, n_estimators="auto", verbose=2, random_state=42
    )
    # X = np.concatenate((X, np.zeros_like(X[:,1]).reshape(-1, 1)), axis=1)
    # 进行特征选择
    boruta_selector.fit(X, y)

    # 查看被选中的特征
    selected_features = boruta_selector.support_
    print("被选中的特征索引：", np.where(selected_features)[0])

    # 如果你的数据是 pandas DataFrame 格式，可以获取对应的特征名称
    # 假设原始特征名称保存在 X.columns 中：
    # selected_feature_names = df.drop('target', axis=1).columns[boruta_selector.support_]
    # print("被选中的特征名称：", selected_feature_names.tolist())

    # 如果需要查看各特征的重要性排名（1 表示最重要）
    ranking = boruta_selector.ranking_
    print("各特征的重要性排名：", ranking)


def cost_sensitive_loss(K=5.0):
    """
    定义一个自定义损失函数。
    参数 K：表示类别 1 与类别 2 混淆时的额外惩罚成本。
    """
    # 定义成本矩阵
    # 注意：这里的矩阵定义中，对角线为 0，其它错误成本为 1，
    # 而当真实类别为1且预测类别为2，或真实类别为2且预测类别为1时成本为 K。
    cost_matrix = tf.constant(
        [[0.0, 1.0, 1.0], [1.0, 0.0, K], [1.0, K, 0.0]], dtype=tf.float32
    )

    def loss_fn(y_true, y_pred):
        # 标准交叉熵损失
        ce_loss = sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

        # 计算每个样本的额外成本
        # 从成本矩阵中取出对应的行，得到 shape (batch_size, n_classes)
        cost_for_sample = tf.gather(cost_matrix, tf.cast(y_true, tf.int32))
        # 计算期望成本：对每个样本，预测概率和对应成本的内积
        extra_cost = tf.reduce_sum(y_pred * cost_for_sample, axis=-1)

        # 最终损失可以是标准交叉熵加上额外的成本项（你可以调整权重）
        total_loss = ce_loss + extra_cost
        return total_loss

    return loss_fn


def dataset_report(X, y, features):
    unique_labels, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    result = ""
    result += f"特征：\n{' '.join(features)}\n"
    result += f"特征总数：{len(features)}\n"
    result += (
        f"包含 NaN 或无穷大（Inf）的行："
        f"{X[np.any(np.isnan(X) | np.isinf(X), axis=1)]}\n"
    )
    result += f"X: {X.shape}\n{X[:5, :]}\n"
    result += f"y: {y.shape}\n{y[:5]}\n"
    result += "y各类别的比例：\n"
    for label, proportion in zip(unique_labels, proportions):
        result += f"类别 {label}: {proportion:.2f}\n"
    return result


def final_model_report(original_pred, meta_label, target_label, threshold=None):
    """
    评估复合模型表现。

    当 meta_label_class 为 'class' 时，meta_label 的取值：
        1 表示原始预测正确，
        0 表示原始预测错误。

    构造复合模型预测方法：
        - 若 meta_label 为 1，则复合模型预测等于原始预测；
        - 若 meta_label 为 0，则认为原始预测错误，复合模型预测取反（假设二分类标签为 0 和 1）。

    参数：
        original_pred: array-like，原始模型的预测结果
        meta_label: array-like，元标签（当 meta_label_class 为 'class' 时为 0 或 1）
        target_label: array-like，真实的目标标签
        meta_label_class: str，默认 'class'，可根据不同情况扩展其他逻辑

    返回：
        report: dict，包含原始模型准确率、复合模型准确率、
                复合模型的分类报告以及混淆矩阵
    """

    # 确保输入为 numpy 数组
    original_pred = original_pred
    meta_label = meta_label
    target_label = target_label

    # 根据 meta_label_class 构造复合模型预测
    if threshold is None:
        # 对于二分类问题，若 meta_label 为 1 保留原始预测，
        # meta_label 为 0 则取反（即 1 - original_pred）
        composite_pred = np.where(meta_label == 1, original_pred, 3)
    else:
        # 如需处理其他类型的 meta_label，可在此扩展
        composite_pred = np.where(meta_label < threshold, 3, original_pred)

    # 计算原始模型和复合模型的准确率
    original_accuracy = accuracy_score(target_label, original_pred)
    composite_accuracy = accuracy_score(target_label, composite_pred)
    # 获取复合模型的分类报告和混淆矩阵
    composite_report = classification_report(target_label, composite_pred, zero_division=0)
    composite_conf_matrix = confusion_matrix(target_label, composite_pred)

    # 构造报告字典
    report = {
        "original_accuracy": original_accuracy,
        "composite_accuracy": composite_accuracy,
        "composite_classification_report": composite_report,
        "composite_confusion_matrix": composite_conf_matrix
    }
    # print("Original Accuracy: {:.4f}".format(report["original_accuracy"]))
    # print("Composite Accuracy: {:.4f}".format(report["composite_accuracy"]))
    print("\nComposite Classification Report:")
    print(report["composite_classification_report"])


def detailed_error_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # 假设类别顺序为 [0, 1, 2]，混淆矩阵的各元素解释为：
    # cm[i, j] 表示真实类别 i 被预测为类别 j 的样本数。

    # 计算各类别的样本总数（真实标签）
    total_class_1 = np.sum(cm[1, :])
    total_class_2 = np.sum(cm[2, :])

    # 对于真实类别1：
    # - 错误预测为2的概率 = cm[1, 2] / total_class_1
    # - 错误预测为0的概率 = cm[1, 0] / total_class_1
    prob_1_pred_2 = cm[1, 2] / total_class_1 if total_class_1 > 0 else 0
    prob_1_pred_0 = cm[1, 0] / total_class_1 if total_class_1 > 0 else 0

    # 对于真实类别2：
    # - 错误预测为1的概率 = cm[2, 1] / total_class_2
    # - 错误预测为0的概率 = cm[2, 0] / total_class_2
    prob_2_pred_1 = cm[2, 1] / total_class_2 if total_class_2 > 0 else 0
    prob_2_pred_0 = cm[2, 0] / total_class_2 if total_class_2 > 0 else 0
    report = "\n针对类别1：\n"
    report += "  实际类别1被预测为类别2的概率： {:.2f}%\n".format(prob_1_pred_2 * 100)
    report += "  实际类别1被预测为类别0的概率： {:.2f}%\n".format(prob_1_pred_0 * 100)

    report += "\n针对类别2：\n"
    report += "  实际类别2被预测为类别1的概率： {:.2f}%\n".format(prob_2_pred_1 * 100)
    report += "  实际类别2被预测为类别0的概率： {:.2f}%\n".format(prob_2_pred_0 * 100)
    return report

# def print_model_report(
#         model,
#         X_train=None,
#         y_train=None,
#         X_test=None,
#         y_test=None,
#         features=None,
#         use_arg_to_predict=True,
#         detail_error_report=True,
#         dump_to_txt=True,
#         statistics_samples=100,
#         threshold=0.5
# ):
#     if features:
#         title = " ".join(features) if features else "unknown_features"
#         report = []
#         report.append(f"特征：{title}")
#         report.append(f"特征总数：{len(features)}")
#     if X_train is not None and y_train is not None:
#         report.append("-----------------------训练集---------------------------")
#         # 获取概率
#         # proba_train = model.predict_proba(X_train)
#         # max_proba_train = np.max(proba_train, axis=1)
#         # y_pred_train = np.argmax(proba_train, axis=1)  # 默认预测
#         # # 应用阈值，将低置信度样本归为“类别 3”
#         # y_pred_train_with_uncertain = y_pred_train.copy()
#         # y_pred_train_with_uncertain[max_proba_train < threshold] = 3
#         # # 只评估确定类别的样本（排除类别 3）
#         # mask_train = y_pred_train_with_uncertain != 3
#         # y_train_filtered = y_train[mask_train]
#         # y_pred_train_filtered = y_pred_train_with_uncertain[mask_train]
#
#         report.append("分类报告（排除不确定类别）：\n" +
#                       classification_report(y_train_filtered, y_pred_train_filtered))
#         if detail_error_report:
#             report.append("详细报告（排除不确定类别）：\n" +
#                           detailed_error_report(y_train_filtered, y_pred_train_filtered))
#         # 报告不确定类别的比例
#         uncertain_ratio_train = 1 - mask_train.mean()
#         report.append(f"不确定类别比例：{uncertain_ratio_train:.2%}")
#     if X_test is not None and y_test is not None:
#         report.append("-----------------------测试集---------------------------")
#         # 获取概率
#         proba_test = model.predict_proba(X_test)
#         max_proba_test = np.max(proba_test, axis=1)
#         y_pred_test = np.argmax(proba_test, axis=1)  # 默认预测
#         # 应用阈值，将低置信度样本归为“类别 3”
#         y_pred_test_with_uncertain = y_pred_test.copy()
#         y_pred_test_with_uncertain[max_proba_test < threshold] = 3
#         # 只评估确定类别的样本（排除类别 3）
#         mask_test = y_pred_test_with_uncertain != 3
#         y_test_filtered = y_test[mask_test]
#         y_pred_test_filtered = y_pred_test_with_uncertain[mask_test]
#         report.append("分类报告（排除不确定类别）：\n" +
#                       classification_report(y_test_filtered, y_pred_test_filtered))
#         if detail_error_report:
#             report.append("详细报告（排除不确定类别）：\n" +
#                           detailed_error_report(y_test_filtered, y_pred_test_filtered))
#         # 报告不确定类别的比例
#         uncertain_ratio_test = 1 - mask_test.mean()
#         report.append(f"不确定类别比例：{uncertain_ratio_test:.2%}")
#         # Baseline
#         report.append("----------------------baseline-------------------------")
#         y_pred_baseline = np.random.randint(0, 3, size=y_test.shape)
#         report.append("分类报告：\n" + classification_report(y_test, y_pred_baseline))
#         if detail_error_report:
#             report.append("详细报告：\n" + detailed_error_report(y_test, y_pred_baseline))
#     if dump_to_txt:
#         os.makedirs("reports", exist_ok=True)
#         output_file = "reports/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
#         with open(output_file, "w", encoding="utf-8") as f:
#             for line in report:
#                 f.write(line + "\n")
#
#     print("\n".join(report))
#
#     # print("-----------------------可视化统计---------------------------")
#     # if statistics_samples > 0:
#     #     # 从测试集中随机抽取100个样本
#     #     indices = np.random.choice(len(X_test), statistics_samples, replace=False)
#     #     X_test_sample = X_test[indices]
#     #     y_test_sample = y_test[indices]
#
#     #     # 获取模型在抽取样本上的预测概率
#     #     y_pred_prob_sample = model.predict(X_test_sample)
#
#     #     # 将预测概率转换为类别标签（0, 1 或 2），即取概率最大的索引
#     #     y_pred_sample = np.argmax(y_pred_prob_sample, axis=1)
#
#     #     # 绘制测试集的真实值与预测值对比图（使用散点图，不连接点）
#     #     plt.figure(figsize=(10, 6))
#     #     plt.scatter(
#     #         np.arange(len(y_test_sample)),
#     #         y_test_sample,
#     #         label="real",
#     #         color="blue",
#     #         marker="o",
#     #     )
#     #     plt.scatter(
#     #         np.arange(len(y_pred_sample)),
#     #         y_pred_sample,
#     #         label="predict",
#     #         color="red",
#     #         marker="x",
#     #     )
#
#     #     plt.title("Result")
#     #     plt.xlabel("index")
#     #     plt.ylabel("category")
#     #     plt.legend()
#     #     plt.show()
