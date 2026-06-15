# 第6章 决策树

## 学习目标
- 能够解释决策树“递归划分特征空间”的建模思想，并画出基本树结构。
- 能够根据信息熵、信息增益、增益率、基尼指数比较候选划分特征。
- 能够区分 ID3、C4.5、CART 的核心差异及适用场景。
- 能够说明预剪枝和后剪枝的作用，并结合过拟合现象给出调参策略。

## 关键词
- 决策树（Decision Tree）
- 信息熵（Entropy）
- 信息增益（Information Gain）
- 增益率（Gain Ratio）
- 基尼指数（Gini Index）
- ID3 / C4.5 / CART
- 预剪枝（Pre-pruning）
- 后剪枝（Post-pruning）

## 核心概念与原理
### 关键定义
- **决策树**：以树结构表示条件判断规则的模型。
- **内部节点**：一个特征上的划分条件。
- **叶节点**：分类结果或回归预测值。

### 方法直觉
- 每次选择“最能把类别分开”的特征，递归细分样本集合。
- 直到节点足够“纯”或达到停止条件。

### 与相近方法的区别
- 与线性模型相比：决策树可自然表达非线性与特征交互。
- 与 SVM 相比：决策树解释性更强，但单树稳定性较弱，易过拟合。

## 关键公式与解释
- 信息熵：
\[
H(D)=-\sum_{k=1}^{K}p_k\log_2 p_k
\]
- 信息增益：
\[
Gain(D,A)=H(D)-\sum_{v=1}^{V}\frac{|D_v|}{|D|}H(D_v)
\]
- 基尼指数：
\[
Gini(D)=1-\sum_{k=1}^{K}p_k^2
\]
- 符号解释：\(p_k\) 为第 \(k\) 类比例，\(D_v\) 为按特征 \(A\) 取值 \(v\) 划分后的子集。
- 公式作用：衡量“划分后纯度是否提升”。
- 使用前提：特征可做离散划分或阈值划分。
- 常见误用点：只比较训练集纯度，不验证泛化；连续特征阈值搜索不充分。

## 算法流程 / 方法步骤
1. **初始化根节点**：输入全部训练样本，输出根节点候选特征集；目的为开始划分。
2. **选择最优划分特征**：输入当前节点样本，输出最佳特征/阈值；目的为最大化纯度提升。
3. **生成子节点**：输入划分结果，输出多个子集节点；目的为细化决策边界。
4. **递归建树**：输入子节点样本，输出子树；目的为逐步逼近训练分布。
5. **停止与剪枝**：输入候选树与验证信息，输出最终树；目的为控制复杂度并提升泛化。

## 实践示例（Python/sklearn）
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=4,
    random_state=42
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("tree depth:", clf.get_depth(), "leaves:", clf.get_n_leaves())
```
- 关键参数：`max_depth` 控制模型容量；`min_samples_split` 抑制碎片化划分。
- 观察结果：除准确率外，查看树深和叶子数判断是否过拟合。

## 常见易错点
- 错因：把信息熵、信息增益、增益率混为一谈。纠正建议：先明确“纯度函数”与“选择偏好”。
- 错因：仅追求训练集 100% 准确。纠正建议：使用验证集和剪枝控制树复杂度。
- 错因：连续特征直接离散化过粗。纠正建议：通过候选阈值搜索最优切分点。
- 错因：类别不平衡只看准确率。纠正建议：增加 `class_weight` 并关注召回率/F1。

## 练习
1. **概念题**：ID3、C4.5、CART 各自默认使用哪种划分标准？  
   参考要点：ID3-信息增益；C4.5-增益率；CART-基尼指数（分类）/平方误差（回归）。
2. **理解题**：为什么单棵决策树容易过拟合？  
   参考要点：递归划分过细会拟合噪声，叶节点样本过少导致高方差。
3. **应用题**：给定两个特征，一个信息增益高但取值很多，另一个增益略低但取值少，为什么 C4.5 可能更偏向后者？  
   参考要点：增益率对多取值特征有惩罚，降低“编号型特征”偏好。
4. **综合题（代码参数）**：在示例代码中将 `max_depth` 从 4 调到 10，若训练准确率上升但测试下降，如何解释并改进？  
   参考要点：出现过拟合；可降低深度、提高 `min_samples_leaf`、用剪枝或随机森林。

## 小结
- 决策树通过“局部最优划分 + 递归建树”完成非线性建模。
- 信息增益/增益率/基尼指数是常见节点划分依据。
- 决策树易解释但易过拟合，剪枝与复杂度约束是关键。
- 单树常作为集成学习（随机森林、梯度提升树）的基础模块。
