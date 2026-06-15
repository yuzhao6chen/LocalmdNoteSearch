# 第1章 机器学习概述

## 学习目标
- 能够说明机器学习与传统规则编程的区别，并给出典型应用场景。
- 能够区分监督学习、无监督学习与强化学习的任务目标和输入输出形式。
- 能够解释训练集、验证集、测试集的作用，并描述基本实验流程。
- 能够用“欠拟合-过拟合-泛化”框架分析模型表现。

## 关键词
- 机器学习（Machine Learning, ML）
- 监督学习（Supervised Learning）
- 无监督学习（Unsupervised Learning）
- 强化学习（Reinforcement Learning, RL）
- 泛化（Generalization）
- 经验风险最小化（Empirical Risk Minimization, ERM）
- 过拟合（Overfitting）
- 偏差-方差（Bias-Variance）

## 核心概念与原理
### 关键定义
- **机器学习**：从数据中学习规律，并在未见样本上进行预测或决策的方法。
- **样本、特征、标签**：样本是观测对象；特征是输入变量；标签是监督任务中的目标输出。
- **模型**：由参数化函数表示，如线性模型、树模型、神经网络等。

### 方法直觉
- 传统程序强调“人工写规则”，机器学习强调“让数据反推规则”。
- 数据越有代表性，模型越可能学到可泛化的规律，而不是记忆训练样本。

### 与相近方法的区别
- 与统计建模：机器学习更强调预测性能和工程可扩展性。
- 与深度学习：深度学习是机器学习子集，主要使用多层神经网络进行表示学习。

## 关键公式与解释
- 经验风险最小化目标：
\[
\hat{f}=\arg\min_{f\in\mathcal{F}}\frac{1}{m}\sum_{i=1}^{m}\ell\big(f(x_i),y_i\big)
\]
- 符号解释：
  - \(m\)：训练样本数；
  - \(f\)：待学习模型；
  - \(\ell(\cdot)\)：损失函数，衡量预测误差。
- 公式作用：通过最小化平均损失学习参数。
- 使用前提：训练数据应与未来数据分布近似一致。
- 常见误用点：只优化训练误差，不检查验证/测试表现，容易过拟合。

## 算法流程 / 方法步骤
1. **问题定义**：输入业务目标，输出任务类型（分类/回归/聚类）；目的是明确学习目标。
2. **数据准备**：输入原始数据，输出可训练数据集；目的是清洗、编码、划分数据。
3. **模型训练**：输入训练集与算法，输出已训练模型；目的是学习参数。
4. **模型评估**：输入验证/测试集，输出指标（如准确率、F1、RMSE）；目的是估计泛化能力。
5. **迭代优化**：输入误差分析结果，输出改进策略；目的是调参与特征改进。

## 实践示例（Python/sklearn）
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, pred))
```
- 关键参数：`test_size` 控制评估可靠性；`max_iter` 影响是否收敛。
- 观察结果：看 `accuracy`，并可进一步看混淆矩阵判断错分模式。

## 常见易错点
- 错因：把测试集用于调参。纠正建议：固定测试集，只在验证集上调参。
- 错因：只看单一指标（如准确率）。纠正建议：结合任务使用 F1、召回率或 AUC。
- 错因：忽略数据分布偏移。纠正建议：按时间或场景做分层评估。
- 错因：把相关性当因果。纠正建议：结论中明确“预测关系”与“因果关系”边界。

## 练习
1. **概念题**：监督学习与无监督学习的输入输出有什么差异？  
   参考要点：监督学习有标签，目标是学习映射；无监督学习无标签，目标是发现结构。
2. **理解题**：为什么“训练准确率高”不等于“模型好”？  
   参考要点：可能记忆训练数据，泛化能力差；需看验证/测试表现。
3. **应用题**：某二分类任务类别严重不平衡，为什么不能只看准确率？  
   参考要点：模型可能全预测多数类仍高准确；应看召回率、F1、PR 曲线。
4. **综合题（参数分析）**：将示例中的 `test_size` 从 0.2 改为 0.5，会对训练稳定性和评估方差产生什么影响？  
   参考要点：训练样本减少可能降性能；测试更大评估更稳定；需折中。

## 小结
- 机器学习核心是“用数据学规则”，目标是泛化而非记忆。
- 实验流程至少包含数据划分、训练、验证、测试四环节。
- 经验风险最小化是多数算法的统一视角。
- 过拟合控制与评估设计和模型选择同等重要。
