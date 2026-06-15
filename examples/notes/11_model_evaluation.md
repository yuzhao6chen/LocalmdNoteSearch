# 第11章 模型评估与调参

## 学习目标
- 能够解释训练集、验证集、测试集的角色，并设计规范评估流程。
- 能够根据任务类型选择合适指标，而不是机械使用准确率。
- 能够使用交叉验证与学习曲线识别欠拟合、过拟合和数据问题。
- 能够基于网格搜索或随机搜索执行可复现调参，并避免数据泄漏。

## 关键词
- 训练集/验证集/测试集（Train/Validation/Test）
- 交叉验证（Cross Validation, CV）
- 混淆矩阵（Confusion Matrix）
- 精确率/召回率（Precision/Recall）
- F1 分数（F1-score）
- ROC-AUC / PR-AUC
- 偏差-方差（Bias-Variance）
- 超参数优化（Hyperparameter Tuning）

## 核心概念与原理
### 关键定义
- **模型评估**：估计模型在未见数据上的表现。
- **超参数**：训练前设定、不由梯度直接学习的参数，如 `max_depth`、`C`、`learning_rate`。
- **泛化误差**：模型在新样本上的平均误差。

### 方法直觉
- 训练指标高只能说明“记住了训练数据”，不能保证泛化。
- 评估核心是“模拟未来数据输入场景”。

### 与相近方法的区别
- 与离线训练日志不同：模型评估强调外部验证和可对比性。
- 与 A/B 测试相比：离线评估成本低、迭代快，但不完全等价线上收益。

## 关键公式与解释
- 准确率：
\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]
- 精确率与召回率：
\[
Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN}
\]
- F1 分数：
\[
F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}
\]
- 符号解释：`TP/TN/FP/FN` 分别为真正例、真反例、假正例、假反例。
- 公式作用：在不同业务成本下刻画模型错误类型。
- 使用前提：明确正负类定义与阈值策略。
- 常见误用点：类别极不平衡时只看准确率，会高估模型效果。

## 算法流程 / 方法步骤
1. **数据划分**：输入全量数据，输出训练/验证/测试集；目的是隔离调参与最终评估。
2. **基线训练**：输入训练集与初始参数，输出基线模型；目的是建立对照。
3. **交叉验证调参**：输入候选参数网格，输出最优参数组合；目的是稳健选择超参数。
4. **阈值与指标分析**：输入验证概率输出，输出 ROC/PR 与阈值方案；目的是匹配业务目标。
5. **最终测试**：输入固定模型与测试集，输出最终报告；目的是给出可复现结论。

## 实践示例（Python/sklearn）
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]

print("best params:", search.best_params_)
print("test roc_auc:", roc_auc_score(y_test, proba))
print(classification_report(y_test, pred))
```
- 关键参数：`scoring="f1"` 体现不平衡任务目标；`StratifiedKFold` 保持类别分布。
- 观察结果：同时看 `F1`、`ROC-AUC` 与分类报告，不只看单一指标。

## 常见易错点
- 错因：先看测试集再改参数。纠正建议：测试集只在最终一次评估使用。
- 错因：预处理先在全数据 `fit`。纠正建议：把预处理放进 `Pipeline`，避免数据泄漏。
- 错因：默认阈值 0.5 直接上线。纠正建议：基于验证集按业务成本选阈值。
- 错因：只做单次随机划分。纠正建议：使用交叉验证减少偶然性。

## 练习
1. **概念题**：为什么测试集不能用于调参？  
   参考要点：会引入信息泄漏，导致测试分数乐观偏差。
2. **理解题**：医疗筛查中为什么常优先优化召回率而非准确率？  
   参考要点：漏诊代价高，应减少假负例。
3. **应用题**：二分类样本比例 95:5，模型准确率 95%，你如何判断模型是否有价值？  
   参考要点：需看少数类召回率、F1、PR-AUC，可能只是全预测多数类。
4. **综合题（参数分析）**：将示例中的 `scoring` 从 `f1` 改成 `accuracy`，最优 `C` 可能变化吗？为什么？  
   参考要点：会变化；优化目标改变导致选出的参数更偏向多数类总体正确率。

## 小结
- 模型评估本质是估计泛化，而不是美化训练分数。
- 指标选择必须绑定业务错误成本和数据分布。
- 交叉验证与 `Pipeline` 是稳定评估和防泄漏的基础实践。
- 调参应可复现、可解释，并以最终测试报告收敛结论。
