# 机器学习模型库

欢迎来到我的机器学习模型库！这里有一些常见的机器学习算法，每个算法都有一个简短的说明和相关示例的链接。

## 线性回归 (Linear Regression)

**实现过程：** 线性回归用于预测连续数值。它建立了输入特征和输出之间的线性关系。我们使用训练数据来学习权重和偏差，以拟合最佳拟合线。这条线将帮助我们对新数据进行预测。

**简单描述：** 这个示例是关于如何使用线性回归模型来预测红酒的品质。我们基于不同的特征（如酒精含量、酸度等）来建立一个模型，以预测红酒的品质评分。

#### [进入线性回归示例](https://github.com/frenkiefang/machine_learning/blob/main/linear_regression.ipynb)

## 逻辑回归 (Logistic Regression)

**实现过程：** 逻辑回归是一种用于解决分类问题的模型。在这个示例中，我们使用逻辑回归来预测航班满意度。我们基于不同的特征，如航班的服务质量、延误情况、空间舒适度等，以及客户的调查问卷回答，建立了一个模型，将满意度分为两个类别：满意和不满意。

**简单描述：** 这个示例演示了如何使用逻辑回归模型来进行二分类预测，以判断乘客对航班的整体满意度。我们将客户的反馈数据与航班满意度评分相关联，以便预测满意度是否高于或低于特定阈值。

#### [进入线性回归示例](https://github.com/frenkiefang/machine_learning/blob/main/logistic_regression.ipynb)

## 支持向量机（Support Vector Machine）

**实现过程：** 支持向量机（SVM）是一种强大的监督学习模型，可用于分类任务。在这个示例中，我们使用相同的航班满意度数据集，探讨了线性和非线性分类问题。首先，我们使用线性SVM来尝试将航班满意度分为两个类别。然后，我们引入非线性SVM，使用不同的内核函数，如RBF核，来处理更复杂的分类问题。

**简单描述：** 这个示例展示了如何使用SVM模型来进行线性和非线性分类，以判断乘客对航班的满意度。我们将不同的SVM方法应用于相同的航班满意度数据集，以探讨线性和非线性分类的差异。

#### [进入支持向量机示例](https://github.com/frenkiefang/machine_learning/blob/main/svm.ipynb)

## 决策树（Decision Tree）

**什么是决策树：** 决策树是一种机器学习算法，通常用于分类和回归问题。它是一种树状结构，用于帮助做出决策或预测结果。在决策树中，每个节点代表一个属性或特征，每个分支代表一个可能的决策或结果，而叶子节点代表最终的分类或回归结果。通过沿着树从根节点到叶子节点的路径进行导航，可以根据输入数据的特征来做出决策或进行预测。

**决策树如何生成：** 

![实例图](./images/decision_tree.png)

1. **根节点 (Gender = 'M')：**

   树的起始点，根据性别特征分割数据。如果年龄小于30岁，向左分支；否则，向右分支。

2. **第二层的节点：**

   **Salary< 500**：如果年龄小于30岁，判断Salary是否小于500。如果是，进入左边分支，否则进入右边分支。

   **Salary < 1000**：如果年龄不小于30岁，检查工资是否小于1000。如果是，进入左边分支，表示对于年龄不小于30岁且工资不小于1000的男性，进入右边分支。

3. **叶子节点：**

   **A, C, D**：这个叶子节点代表了年龄小于30岁且工资低于500的的分类结果。

   **B, F**：这些叶子节点代表了年龄小于30岁但工资不低于于500的男性的分类结果。

   **E**：这个叶子节点代表了年龄不小于30岁且工资小于1000的男性的分类结果。

   **G, H**：这些叶子节点代表了年龄不小于30岁且工资不小于1000的男性的分类结果。



***

欢迎随时查看这些示例以深入了解这些机器学习算法的实现方式。如果您有任何问题或建议，欢迎与我联系！

