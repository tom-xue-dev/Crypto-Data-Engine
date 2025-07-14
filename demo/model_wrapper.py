from datetime import datetime

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # print("using GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], "GPU")


class ModelWrapper:
    def __init__(self, name):
        self.model = None
        self.name = name
        self._initialize()

    def _initialize(self):
        if self.name == "MLP":
            from tensorflow.keras import Sequential, Input
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
            from tensorflow.keras.losses import SparseCategoricalCrossentropy
            from tensorflow.keras.regularizers import l2
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
            self.model = Sequential(
                [
                    Input(shape=(self.train_feature.shape[1],)),
                    Dense(
                        512,
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(0.001),
                    ),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.01),
                    Dropout(0.2),
                    Dense(
                        200,
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(0.001),
                    ),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.01),
                    Dropout(0.2),
                    Dense(
                        200,
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(0.001),
                    ),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.01),
                    Dropout(0.2),
                    Dense(
                        200,
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(0.001),
                    ),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.01),
                    Dropout(0.2),
                    Dense(
                        128,
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(0.001),
                    ),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.01),
                    Dense(1, activation="linear"),
                ]
            )
            early_stop = EarlyStopping(
                monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=1
            )
            log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = TensorBoard(
                log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True
            )
            self.callbacks = [early_stop, reduce_lr, tensorboard_callback]
            # self.model.compile(
            #     optimizer=Adam(learning_rate=0.001),
            #     loss=SparseCategoricalCrossentropy(from_logits=True),
            #     metrics=["accuracy"],
            # )
            self.model.compile(
                optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
                loss='mse',
                metrics=['mae']
            )

        elif self.name == "RF":

            # class_weight = {0: 2, 1: 1, 2: 1}

            self.model = RandomForestClassifier(
                n_estimators=300,
                min_samples_split=2,
                min_samples_leaf=2,
                max_depth=15,
                max_features='log2',  # 改为'sqrt'，每次只考虑部分特征
                max_samples=0.5,  # 适当降低样本抽取比例
                random_state=42,
                class_weight='balanced_subsample',
                bootstrap=True,
                n_jobs=16,
            )

            # clf = DecisionTreeClassifier(
            #     criterion='entropy',
            #     max_features='sqrt',
            #     class_weight='balanced',
            # )
            # self.model = BaggingClassifier(
            #     estimator=clf,
            #     n_estimators=300,
            #     max_samples=0.40962,
            #     max_features=1.,
            #     n_jobs=10
            # )

            # 调整后的随机森林基分类器
            # clf = RandomForestClassifier(
            #     n_estimators=30,  # 增加到 30 棵树
            #     criterion='gini',
            #     max_depth=20,  # 深度 20
            #     min_samples_split=5,
            #     min_samples_leaf=2,
            #     bootstrap=True,
            #     max_features='sqrt',
            #     class_weight={0: 1, 1: 2, 2: 2},  # 更激进的权重
            #     n_jobs=-1
            # )
            #
            # self.model = BaggingClassifier(
            #     estimator=clf,
            #     n_estimators=10,  # 减少到 10 个基分类器
            #     max_samples=0.7,  # 70% 数据
            #     max_features=0.8,  # 80% 特征
            #     n_jobs=-1
            # )
        elif self.name == "XGB":

            import xgboost as xgb

            def custom_objective(preds, dtrain):
                """
                自定义目标函数：在 softmax 交叉熵的基础上，
                针对类别 1 与 2 的互相误判，乘以额外的惩罚因子 lambda_cost。
                """
                labels = dtrain.get_label().astype(int)
                num_class = 3
                # 重塑预测值为 (n_samples, num_class)
                preds = preds.reshape(-1, num_class)

                # 计算 softmax 概率
                max_pred = np.max(preds, axis=1, keepdims=True)
                exp_preds = np.exp(preds - max_pred)
                sum_exp = np.sum(exp_preds, axis=1, keepdims=True)
                p = exp_preds / sum_exp  # 概率矩阵，形状 (n, num_class)

                # 定义成本矩阵：cost_matrix[true_label, predicted_label]
                # 默认误判成本为 1，只有当 true_label 与 predicted_label 为 1 与 2 互换时，成本加大
                cost_matrix = np.ones((num_class, num_class))
                lambda_cost = 5.0  # 针对 1 和 2 之间误判的额外惩罚因子
                cost_matrix[1, 2] = lambda_cost
                cost_matrix[2, 1] = lambda_cost

                # 初始化梯度和 Hessian
                grad = np.zeros_like(p)
                hess = np.zeros_like(p)
                for i in range(len(labels)):
                    t = labels[i]
                    for j in range(num_class):
                        indicator = 1.0 if j == t else 0.0
                        # 对应类别 j 的梯度：额外乘上成本因子
                        grad[i, j] = cost_matrix[t, j] * (p[i, j] - indicator)
                        # Hessian：这里采用 p*(1-p) 的形式，同样乘上成本因子
                        hess[i, j] = cost_matrix[t, j] * p[i, j] * (1 - p[i, j])

                return grad.reshape(-1), hess.reshape(-1)

            # --- 定义自定义评估指标 ---
            def custom_eval_metric(preds, dtrain):
                """
                自定义评估指标：计算带有成本惩罚的平均误差。
                对于每个样本，误差为 cost_matrix[true, pred]（理想情况下为 1，1-2 或 2-1 则为 lambda_cost）。
                """
                labels = dtrain.get_label().astype(int)
                num_class = 3
                preds = preds.reshape(-1, num_class)
                pred_labels = np.argmax(preds, axis=1)

                cost_matrix = np.ones((num_class, num_class))
                lambda_cost = 5.0
                cost_matrix[1, 2] = lambda_cost
                cost_matrix[2, 1] = lambda_cost

                total_cost = 0.0
                for i in range(len(labels)):
                    total_cost += cost_matrix[labels[i], pred_labels[i]]
                avg_cost = total_cost / len(labels)
                return 'cost', avg_cost

            self.model = xgb.XGBClassifier(
                n_estimators=500,
                objective="multi:softmax",
                num_class=3,
                random_state=42,
                use_label_encoder=False,
            )
        elif self.name == "LGB":

            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=200, random_state=42
            )
        else:
            raise ValueError(f"No such model: {self.name}")

    def train(self,train_feature,train_label):
        if self.name == "MLP":
            history = self.model.fit(
                train_feature,
                train_label,
                epochs=300,
                batch_size=256,
                validation_split=0.2,
                callbacks=self.callbacks,
            )

        elif self.name == "RF":
            self.model.fit(train_feature, train_label)
        elif self.name == "XGB":
            self.model.fit(self.train_feature, self.train_label)
        elif self.name == "LGB":
            self.model.fit(self.train_feature, self.train_label)
        else:
            raise ValueError(f"No such model: {self.name}")

    def set_model_params(self, params):
        """
        根据给定的超参数字典，重新构造一个 RandomForestClassifier 模型，并赋值给 self.model。

        参数：
            params (dict): 包含 RandomForestClassifier 超参数的字典，
                           例如：{
                               'n_estimators': 200,
                               'min_samples_split': 2,
                               'min_samples_leaf': 2,
                               'max_depth': 10,
                               'max_features': 'sqrt',
                               'max_samples': 0.7,
                               'random_state': 42,
                               'class_weight': 'balanced_subsample',
                               'bootstrap': True,
                               'n_jobs': 16
                           }
        """
        if not isinstance(params, dict):
            raise ValueError("参数 'params' 必须是一个字典")
        self.model = RandomForestClassifier(**params)

    def save(self, label=None):

        import joblib
        import os

        os.makedirs("models", exist_ok=True)
        path = f"models/{self.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if label:
            path += f"-{label}"
        joblib.dump(self.model, f"{path}.pkl")


class ModelAnalyzer:
    def __init__(self, name, model, train_feature, train_label, test_feature, test_label):
        self.name = name
        self.model = model
        self.train_feature = train_feature
        self.train_label = train_label
        self.test_feature = test_feature
        self.test_label = test_label

    def generate_predictions(self, prediction_mode='class', threshold=None):
        """
        计算预测结果，应用不确定性阈值，并返回预测值和不确定性比例。
        :param threshold: 置信度阈值，低于此值的预测会归为“不确定类别”
        :param prediction_mode:返回的数据格式
        :return: 训练集结果，测试集结果，训练集不确定概率（可选），测试集不确定概率（可选）
        """
        # 获取预测概率
        if self.name == "RF":
            self.model: RandomForestClassifier
            # proba 返回的是一个二维数组，axis=0是行（对应pandas里的每一行） axis=1是列，对应的每一行的不同label的概率
            train_predict_matrix = self.model.predict_proba(self.train_feature)
            sample_1_prob_train = train_predict_matrix[:, 1]
            train_prob = np.max(train_predict_matrix, axis=1)

            train_class = np.argmax(train_predict_matrix, axis=1)  # 训练集预测的类别结果

            test_predict_matrix = self.model.predict_proba(self.test_feature)
            test_prob = np.max(test_predict_matrix, axis=1)
            sample_1_prob_test = test_predict_matrix[:, 1]
            test_class = np.argmax(test_predict_matrix, axis=1)  # 测试集预测的类别结果

            if prediction_mode == "prob":
                return sample_1_prob_train, sample_1_prob_test, None, None
            elif prediction_mode == "class":
                if threshold:
                    # 训练集的阈值筛选结果
                    train_pred_with_uncertain = train_class.copy()
                    train_pred_with_uncertain[train_prob < threshold] = 3
                    train_mask = train_pred_with_uncertain != 3
                    train_uncertain_ratio = 1 - train_mask.mean()
                    # 测试集阈值筛选结果
                    test_pred_with_uncertain = test_class.copy()
                    test_pred_with_uncertain[test_prob < threshold] = 3
                    test_mask = test_pred_with_uncertain != 3
                    test_uncertain_ratio = 1 - test_mask.mean()

                    return train_pred_with_uncertain, test_pred_with_uncertain, train_uncertain_ratio, test_uncertain_ratio
                else:
                    return train_class, test_class, None, None
            else:
                print(f"invalid args for param prediction model, plese choose 'prob' or 'class'")
        elif self.name == "MLP":
            if prediction_mode == "class":
                train_logits = self.model.predict(self.train_feature)  # 预测的是未归一化的 logits
                train_predictions = np.argmax(train_logits, axis=1)  # 获取最大概率的类别索引
                test_logits = self.model.predict(self.test_feature)
                test_predicitons = np.argmax(test_logits, axis=1)
                return train_predictions, test_predicitons, None, None
            else:
                # # 获取 logits
                # train_logits = self.model.predict(self.train_feature)
                # test_logits = self.model.predict(self.test_feature)
                #
                # # 使用 Softmax 转换为概率
                # train_probs = tf.nn.softmax(train_logits).numpy()
                # test_probs = tf.nn.softmax(test_logits).numpy()
                # # 获取类别 1 的概率（假设类别索引 1 对应的是目标类别）
                # train_class1_probs = train_probs[:, 1]  # 第 1 列表示类别 1 的概率
                # test_class1_probs = test_probs[:, 1]
                # # 获取预测类别
                # train_predictions = np.argmax(train_probs, axis=1)
                # test_predictions = np.argmax(test_probs, axis=1)
                # # 返回类别 1 的概率
                # return train_class1_probs, test_class1_probs, None, None
                # 获取预测值（回归任务下直接输出连续值）
                train_preds = self.model.predict(self.train_feature)
                test_preds = self.model.predict(self.test_feature)

                # 如果需要后续处理，比如反归一化，则在这里添加相应代码

                # 返回预测值
                return train_preds, test_preds, None, None

    def print_report(self, prediction_mode='class', threshold=None, detail_error_report=True):
        """
        打印分类报告，包括训练集和测试集的分类准确性，以及不确定类别的比例（如果适用）

        :param prediction_mode: 返回类别 ('class') 或 概率 ('prob')
        :param threshold: 置信度阈值（仅用于 'class' 模式）
        :param detail_error_report: 是否打印详细错误报告
        """
        report = ["********** 模型评估报告 **********"]

        # 获取预测结果
        train_pred, test_pred, train_uncertain_ratio, test_uncertain_ratio = self.generate_predictions(
            prediction_mode=prediction_mode, threshold=threshold
        )

        if threshold is not None:
            # 过滤掉不确定类别 3
            mask_train = train_pred != 3
            y_train_filtered = self.train_label[mask_train]
            y_pred_filtered = train_pred[mask_train]
            # 训练集分类报告
            report.append("\n---------------- 训练集 ----------------")
            report.append("分类报告（排除不确定类别）：\n" +
                          classification_report(y_train_filtered, y_pred_filtered, zero_division=0))
            # 计算不确定类别的比例
            uncertain_ratio_train = 1 - mask_train.mean()
            report.append(f"不确定类别比例（训练集）: {uncertain_ratio_train:.2%}")

            mask_test = test_pred != 3
            test_filter = self.test_label[mask_test]
            test_pred_filter = test_pred[mask_test]
            report.append("\n---------------- 测试集 ----------------")
            report.append("分类报告（排除不确定类别）：\n" +
                          classification_report(test_filter, test_pred_filter, zero_division=0))
            # 计算不确定类别的比例
            uncertain_ratio_test = 1 - mask_test.mean()
            report.append(f"不确定类别比例（训练集）: {uncertain_ratio_test:.2%}")

        else:
            # 训练集报告
            report.append("\n---------------- 训练集 ----------------")
            report.append("分类报告：\n" +
                          classification_report(self.train_label, train_pred, zero_division=0))

            # 测试集报告
            report.append("\n---------------- 测试集 ----------------")
            report.append("分类报告：\n" +
                          classification_report(self.test_label, test_pred, zero_division=0))

        # 打印报告
        print("\n".join(report))
