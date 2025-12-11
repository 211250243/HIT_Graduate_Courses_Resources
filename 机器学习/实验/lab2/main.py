import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

# 全局设置（Windows 常见中文字体：Microsoft YaHei 或 SimHei）
plt.rcParams['font.family'] = ['Microsoft YaHei']  # 或者 ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块

# --- 核心函数 ---

def sigmoid(z):
    """
    计算 Sigmoid 函数
    """
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta, lambda_=0):
    """
    计算成本函数 (对数似然)
    :param X: 特征矩阵 (m, n+1)
    :param y: 标签向量 (m, 1)
    :param theta: 参数 (n+1, 1)
    :param lambda_: 正则化参数
    :return: 成本 J
    """
    m = len(y)
    if m == 0:
        return 0

    h = sigmoid(X @ theta) # @ 矩阵乘法

    # 防止 log(0)
    h = np.clip(h, 1e-10, 1 - 1e-10) # clip 限制数值范围(0,1)

    # 成本
    J = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) # 交叉熵损失函数：L = -[y*log(p) + (1-y)*log(1-p)]

    # L2 正则化
    if lambda_ > 0:
        reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)  # 不惩罚 theta_0（z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ）
        J += reg_term

    return J.item()  # 返回标量


def compute_gradient(X, y, theta, lambda_=0):
    """
    计算梯度
    :param X: 特征矩阵 (m, n+1)
    :param y: 标签向量 (m, 1)
    :param theta: 参数 (n+1, 1)
    :param lambda_: 正则化参数
    :return: 梯度 (n+1, 1)
    """
    m = len(y)
    if m == 0:
        return np.zeros_like(theta)

    h = sigmoid(X @ theta)
    gradient = (1 / m) * (X.T @ (h - y))

    # L2 正则化
    if lambda_ > 0:
        reg_term = (lambda_ / m) * theta
        reg_term[0] = 0  # 不惩罚 theta_0
        gradient += reg_term

    return gradient


def gradient_descent(X, y, alpha=0.01, lambda_=0, num_iters=1000):
    """
    使用梯度下降法训练逻辑回归
    :param X: 特征矩阵 (m, n+1)
    :param y: 标签向量 (m, 1)
    :param alpha: 学习率
    :param lambda_: 正则化参数
    :param num_iters: 迭代次数
    :return: 训练后的 theta, 成本历史
    """
    m, n = X.shape
    # 初始化 theta
    theta = np.zeros((n, 1))
    cost_history = []

    for i in range(num_iters):
        cost = compute_cost(X, y, theta, lambda_)
        gradient = compute_gradient(X, y, theta, lambda_)

        theta = theta - alpha * gradient
        cost_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"迭代 {i:5d}/{num_iters} - 成本: {cost:.4f}")

    return theta, cost_history


def predict(X, theta, threshold=0.5):
    """
    使用训练好的参数进行预测
    :param X: 特征矩阵 (m, n+1)
    :param theta: 参数 (n+1, 1)
    :param threshold: 分类阈值
    :return: 预测标签 (m, 1)
    """
    h = sigmoid(X @ theta)
    return (h >= threshold).astype(int)


def calculate_accuracy(y_true, y_pred):
    """
    计算准确率
    """
    return np.mean(y_true == y_pred) * 100


def add_intercept(X):
    """
    在 X 矩阵的第一列添加偏置项 (全 1)
    """
    m = X.shape[0]
    ones = np.ones((m, 1))
    return np.hstack((ones, X)) # hstack 水平堆叠数组


# --- 绘图函数 ---

def plot_decision_boundary(X, y, theta):
    """
    绘制数据点和决策边界
    仅适用于 2D 特征
    """
    plt.figure(figsize=(10, 6))

    # 绘制数据点
    plt.scatter(X[y.flatten() == 0, 1], X[y.flatten() == 0, 2], c='blue', label='类别 0')
    plt.scatter(X[y.flatten() == 1, 1], X[y.flatten() == 1, 2], c='red', label='类别 1')

    # 绘制决策边界
    # 决策边界是 theta_0 + theta_1*x1 + theta_2*x2 = 0
    # => x2 = (-theta_0 - theta_1*x1) / theta_2
    if theta is not None and len(theta) == 3:
        # 创建 x1 坐标
        x1_plot = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        # 计算 x2 坐标
        x2_plot = (-theta[0] - theta[1] * x1_plot) / theta[2]

        plt.plot(x1_plot, x2_plot, label='决策边界', color='green', linewidth=2)

    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.legend()
    plt.title('逻辑回归决策边界 (高斯数据)')
    plt.show()


# --- 任务 1: 手工生成高斯数据 ---

def run_task_1():
    print("--- 任务 1: 高斯数据 ---")

    # 1. 生成数据：两个类别，每类 2 个特征。创建两个协方差矩阵相同但均值不同的高斯分布。为了使类条件分布不满足朴素贝叶斯假设，让特征相关
    mean0 = [0, 0]
    cov0 = [[2, 1.5], [1.5, 2]]  # 协方差矩阵，特征 x1 和 x2 正相关（协方差为1.5为正，当 x1 增大时 x2 倾向也增大）

    mean1 = [4, 4]
    cov1 = [[2, 1.5], [1.5, 2]]

    n_samples = 200
    X0 = np.random.multivariate_normal(mean0, cov0, n_samples) # multivariate_normal: 多元正态分布
    y0 = np.zeros((n_samples, 1))

    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    y1 = np.ones((n_samples, 1))

    X_raw = np.vstack((X0, X1)) # vstack 垂直堆叠数组
    y = np.vstack((y0, y1))

    # 2. 准备数据
    X_scaled = StandardScaler().fit_transform(X_raw) # 特征缩放 (即标准化，对于梯度下降很重要)
    X = add_intercept(X_scaled) # 添加偏置项

    # 3. 训练模型 (带 L2 正则化)
    alpha = 0.1
    lambda_ = 1  # L2 正则化参数
    num_iters = 1000

    print(f"开始训练 (无惩罚项)... Lambda={0}")
    theta_no_reg, cost_history_no_reg = gradient_descent(X, y, alpha=alpha, lambda_=0, num_iters=num_iters)

    print(f"\n开始训练 (带 L2 惩罚项)... Lambda={lambda_}")
    theta_l2, cost_history_l2 = gradient_descent(X, y, alpha=alpha, lambda_=lambda_, num_iters=num_iters)

    # 4. 评估
    y_pred_l2 = predict(X, theta_l2)
    accuracy_l2 = calculate_accuracy(y, y_pred_l2)
    print(f"\n带 L2 惩罚项的训练集准确率: {accuracy_l2:.2f}%")

    y_pred_no_reg = predict(X, theta_no_reg)
    accuracy_no_reg = calculate_accuracy(y, y_pred_no_reg)
    print(f"无惩罚项的训练集准确率: {accuracy_no_reg:.2f}%")

    # 5. 绘制结果
    # 绘制带 L2 惩罚的决策边界
    # 我们在 X (带偏置项) 上绘制，但 X 的 1, 2 列是缩放后的特征
    plot_decision_boundary(X, y, theta_l2)

    # 绘制成本历史
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history_no_reg, label=f'无惩罚 (Lambda=0)')
    plt.plot(cost_history_l2, label=f'L2 惩罚 (Lambda={lambda_})')
    plt.xlabel('迭代次数')
    plt.ylabel('成本')
    plt.legend()
    plt.title('成本函数下降曲线')
    plt.show()


# --- 任务 2: UCI 数据集测试 ---

def run_task_2():
    print("\n--- 任务 2: UCI 乳腺癌数据集 ---")

    # 1. 加载数据
    data = load_breast_cancer()
    X_raw = data.data
    y = data.target.reshape(-1, 1)  # (m,) -> (m, 1)

    # 2. 准备数据
    X_scaled = StandardScaler().fit_transform(X_raw) # 特征缩放
    X = add_intercept(X_scaled) # 添加偏置项
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 拆分训练集和测试集

    # 3. 训练模型 (无惩罚项)
    alpha = 0.01
    lambda_0 = 0
    num_iters = 1000

    print(f"开始训练 (无惩罚项)... Lambda={lambda_0}")
    theta_0, cost_history_0 = gradient_descent(X_train, y_train, alpha=alpha, lambda_=lambda_0, num_iters=num_iters)

    # 4. 评估 (无惩罚项)
    y_pred_train_0 = predict(X_train, theta_0)
    acc_train_0 = calculate_accuracy(y_train, y_pred_train_0)
    y_pred_test_0 = predict(X_test, theta_0)
    acc_test_0 = calculate_accuracy(y_test, y_pred_test_0)

    print(f"无惩罚项 - 训练集准确率: {acc_train_0:.2f}%")
    print(f"无惩罚项 - 测试集准确率: {acc_test_0:.2f}%")

    # 5. 训练模型 (带 L2 惩罚项)
    lambda_l2 = 1.0

    print(f"\n开始训练 (带 L2 惩罚项)... Lambda={lambda_l2}")
    theta_l2, cost_history_l2 = gradient_descent(X_train, y_train, alpha=alpha, lambda_=lambda_l2, num_iters=num_iters)

    # 6. 评估 (带 L2 惩罚项)
    y_pred_train_l2 = predict(X_train, theta_l2)
    acc_train_l2 = calculate_accuracy(y_train, y_pred_train_l2)
    y_pred_test_l2 = predict(X_test, theta_l2)
    acc_test_l2 = calculate_accuracy(y_test, y_pred_test_l2)

    print(f"L2 惩罚项 - 训练集准确率: {acc_train_l2:.2f}%")
    print(f"L2 惩罚项 - 测试集准确率: {acc_test_l2:.2f}%")

    # 7. 与 Scikit-learn 比较
    print("\n--- Scikit-learn 模型比较 ---")
    # C 是 lambda 的倒数, C=1/lambda。 sklearn 默认使用 L2 惩罚。
    # 我们使用 liblinear 求解器，它适用于小型数据集

    # 对应 lambda=1.0, C=1.0
    clf_l2 = SklearnLogisticRegression(C=1.0 / lambda_l2, solver='liblinear', penalty='l2')
    clf_l2.fit(X_train[:, 1:], y_train.ravel())  # sklearn 内部处理偏置项
    acc_sklearn_l2 = clf_l2.score(X_test[:, 1:], y_test) * 100
    print(f"Sklearn (L2, C={1.0 / lambda_l2}) - 测试集准确率: {acc_sklearn_l2:.2f}%")

    # 对应 lambda=0 (近似)，我们用一个非常大的 C 来模拟没有正则化
    clf_no_reg = SklearnLogisticRegression(C=1e9, solver='liblinear', penalty='l2')
    clf_no_reg.fit(X_train[:, 1:], y_train.ravel())
    acc_sklearn_no_reg = clf_no_reg.score(X_test[:, 1:], y_test) * 100
    print(f"Sklearn (L2, C=1e9) - 测试集准确率: {acc_sklearn_no_reg:.2f}%")


# --- 主程序 ---
if __name__ == "__main__":
    # 运行任务 1
    run_task_1()

    # 运行任务 2
    run_task_2()