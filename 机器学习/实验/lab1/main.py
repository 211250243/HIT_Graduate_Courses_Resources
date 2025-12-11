# -*- coding: utf-8 -*-

"""
实验：多项式拟合正弦函数
目标：系统探究多项式阶数 M、样本数 N、L2正则化系数 lambda 以及训练参数 alpha, Epochs 对拟合结果的影响。
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# --- 1. 数据生成 ---

def generate_data(num_samples, noise_std):
    """
    生成带有高斯噪声的正弦函数数据
    """
    # 在[0, 1]范围内均匀生成x
    x = np.linspace(0, 1, num_samples)

    # 计算真实的y值：y = sin(2 * pi * x)
    y_true = np.sin(2 * np.pi * x)

    # 添加高斯噪声
    noise = np.random.normal(0, noise_std, num_samples)
    y_noisy = y_true + noise

    return x, y_true, y_noisy


# --- 2. 特征工程 ---

def create_polynomial_features(x, degree):
    """
    为给定的x创建多项式特征矩阵
    """
    num_samples = x.shape[0]
    # 初始化特征矩阵，M+1列（包括x^0）
    X = np.zeros((num_samples, degree + 1))

    for i in range(degree + 1):
        X[:, i] = np.power(x, i)

    return X


# --- 3. 损失函数和梯度 ---

def compute_loss(y_pred, y_true, w, lambda_reg):
    """
    计算均方误差（MSE）损失和L2正则化项
    """
    num_samples = y_true.shape[0]

    # 均方误差
    mse_loss = (1 / (2 * num_samples)) * np.sum(np.square(y_pred - y_true))

    # L2 正则化项
    # 注意：不对偏置项 w[0] (对应 x^0) 进行正则化
    l2_penalty = (lambda_reg / 2) * np.sum(np.square(w[1:]))

    total_loss = mse_loss + l2_penalty
    return total_loss


def compute_gradient(X, y_pred, y_true, w, lambda_reg):
    """
    手动计算损失函数关于权重w的梯度
    """
    num_samples = y_true.shape[0]

    # 预测误差
    error = y_pred - y_true

    # 计算MSE损失的梯度: grad_mse = (1/N) * X^T * (Xw - y)
    grad_mse = (1 / num_samples) * X.T.dot(error)

    # 计算L2正则化项的梯度: grad_l2 = lambda * w
    grad_l2 = lambda_reg * w
    grad_l2[0] = 0  # 偏置项 w[0] 的正则化梯度为0

    # 总梯度
    total_grad = grad_mse + grad_l2
    return total_grad


# --- 4. 梯度下降训练 ---

def train_polynomial_regression(x_train, y_train, degree, learning_rate, lambda_reg, epochs):
    """
    使用梯度下降法训练多项式回归模型
    """

    # 1. 创建特征矩阵
    X_train = create_polynomial_features(x_train, degree)

    # 2. 初始化权重
    w = np.random.randn(degree + 1) * 0.1

    loss_history = []

    start_time = time.time()

    # 3. 梯度下降迭代
    for epoch in range(epochs):
        # a. 计算预测值
        y_pred = X_train.dot(w)

        # b. 计算损失
        loss = compute_loss(y_pred, y_train, w, lambda_reg)
        loss_history.append(loss)

        # c. 计算梯度
        grad = compute_gradient(X_train, y_pred, y_train, w, lambda_reg)

        # d. 更新权重
        w = w - learning_rate * grad

        # 仅在需要时打印损失，避免过高的打印频率
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    end_time = time.time()
    print(f"训练完成. 耗时: {end_time - start_time:.2f} 秒")

    return w, loss_history


# --- 5. 预测和绘图 ---

def predict(x, w, degree):
    """
    使用训练好的权重进行预测
    """
    X = create_polynomial_features(x, degree)
    y_pred = X.dot(w)
    return y_pred


def plot_results(x_data, y_data, y_true_curve, x_curve, y_curve, w, title):
    """
    绘制拟合结果
    """
    plt.figure(figsize=(10, 6))
    # 绘制真实的正弦曲线（用于对比）
    plt.plot(x_curve, y_true_curve, 'g-', label='真实正弦函数 $sin(2\pi x)$', linewidth=2)
    # 绘制带噪声的训练数据点
    plt.scatter(x_data, y_data, facecolors='none', edgecolors='b', s=50, label='训练数据点 (含噪声)')
    # 绘制拟合的多项式曲线
    plt.plot(x_curve, y_curve, 'r-', label=f'拟合曲线 (M={w.shape[0] - 1})', linewidth=2)

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.ylim(-1.5, 1.5)  # 设置y轴范围
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_loss_history(loss_history, title):
    """
    绘制损失下降曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, 'b-')
    plt.title(title)
    plt.xlabel('迭代次数 (Epoch)')
    plt.ylabel('损失 (Loss)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# --- 6. 实验执行 ---

def run_experiment(N, M, lambda_reg, learning_rate, epochs, noise_std=0.15, experiment_id=""):
    """
    封装一个完整的实验流程
    """
    title = f"[{experiment_id}] N={N}, M={M}, $\lambda$={lambda_reg}, $\\alpha$={learning_rate}, Epochs={epochs}"
    print(f"--- 开始实验: {title} ---")

    # 1. 生成数据
    x_data, y_true, y_data = generate_data(N, noise_std)

    # 2. 训练模型
    w, loss_history = train_polynomial_regression(
        x_data, y_data, M, learning_rate, lambda_reg, epochs
    )

    # 3. 准备绘图数据
    x_curve = np.linspace(0, 1, 100)
    y_true_curve = np.sin(2 * np.pi * x_curve)
    y_fit_curve = predict(x_curve, w, M)

    # 4. 绘制结果
    plot_results(x_data, y_data, y_true_curve, x_curve, y_fit_curve, w, title)
    plot_loss_history(loss_history, f"损失曲线: {title}")

    # 5. 打印最终的权重
    print(f"最终权重 (w):")
    for i, wi in enumerate(w):
        print(f"  w_{i}: {wi:.4f}")
    print("--------------------------------\n")


if __name__ == "__main__":
    # --- 实验设置：探究不同参数对拟合结果的影响 ---

    # 设定高阶模型（M=9）的默认训练参数，确保充分收敛以展示过拟合
    HIGH_M_LR = 0.05
    HIGH_M_EPOCHS = 1000000

    # 设定低阶模型（M<9）或大数据量（N>10）的默认训练参数，确保稳定
    STABLE_LR = 0.01
    STABLE_EPOCHS = 500000

    # -------------------------------------------------------------------
    # 实验组一：多项式阶数 M 的影响 (固定 N=10, lambda=0)
    # -------------------------------------------------------------------

    # 1.1 欠拟合 (M=3)
    run_experiment(N=10, M=3, lambda_reg=0.0, learning_rate=STABLE_LR, epochs=STABLE_EPOCHS, experiment_id="1.1 M=3")

    # 1.2 适度拟合 (M=5)
    run_experiment(N=10, M=5, lambda_reg=0.0, learning_rate=STABLE_LR, epochs=STABLE_EPOCHS, experiment_id="1.2 M=5")

    # 1.3 严重过拟合 (M=9) - 设定为所有高阶实验的基线
    run_experiment(N=10, M=9, lambda_reg=0.0, learning_rate=HIGH_M_LR, epochs=HIGH_M_EPOCHS,
                   experiment_id="1.3 M=9 (基线)")

    # -------------------------------------------------------------------
    # 实验组二：样本数 N 的影响 (固定 M=9, lambda=0)
    # -------------------------------------------------------------------

    # 2.1 过拟合 (N=10) - 基线 (同 1.3)
    # 不重复运行，直接参考 1.3 结果

    # 2.2 振荡减轻 (N=30)
    run_experiment(N=30, M=9, lambda_reg=0.0, learning_rate=STABLE_LR, epochs=STABLE_EPOCHS, experiment_id="2.2 N=30")

    # 2.3 良好拟合 (N=100)
    run_experiment(N=100, M=9, lambda_reg=0.0, learning_rate=STABLE_LR, epochs=STABLE_EPOCHS, experiment_id="2.3 N=100")

    # -------------------------------------------------------------------
    # 实验组三：训练超参数 ($\lambda, \alpha, Epochs$) 的影响 (固定 N=10, M=9)
    # -------------------------------------------------------------------

    # --- 子实验 A: L2 正则化系数 lambda 的影响 ---

    # 3.A.1 无正则化 (lambda=0.0) - 基线 (同 1.3/2.1)
    # 不重复运行，直接参考 1.3 结果

    # 3.A.2 弱正则化 (lambda=0.0005)
    run_experiment(N=10, M=9, lambda_reg=0.0005, learning_rate=HIGH_M_LR, epochs=HIGH_M_EPOCHS,
                   experiment_id="3.A.2 $\lambda$=0.0005")

    # 3.A.3 强正则化 (lambda=0.005)
    run_experiment(N=10, M=9, lambda_reg=0.005, learning_rate=HIGH_M_LR, epochs=HIGH_M_EPOCHS,
                   experiment_id="3.A.3 $\lambda$=0.005")

    # --- 子实验 B: 学习率 alpha ($\alpha$) 的影响 ---

    # 3.B.1 学习率过低 (alpha=0.001) - 慢速收敛
    # Note: 降低学习率后，为了确保能看出差异，我们只运行 10万轮 (相对 1M 轮算少)
    run_experiment(N=10, M=9, lambda_reg=0.0, learning_rate=0.001, epochs=100000,
                   experiment_id="3.B.1 $\\alpha$=0.001 (慢速)")

    # 3.B.2 学习率正常 (alpha=0.05) - 基线 (同 1.3)
    # 不重复运行，直接参考 1.3 结果

    # 3.B.3 学习率过高 (alpha=0.5) - 发散/剧烈振荡
    # Note: 过高的学习率会在少数轮数内发散，减少epochs以节省时间
    run_experiment(N=10, M=9, lambda_reg=0.0, learning_rate=0.5, epochs=5000,
                   experiment_id="3.B.3 $\\alpha$=0.5 (发散)")

    # --- 子实验 C: 训练轮数 Epochs 的影响 ---

    # 3.C.1 训练轮数过少 (Epochs=5000) - 欠收敛
    run_experiment(N=10, M=9, lambda_reg=0.0, learning_rate=HIGH_M_LR, epochs=5000,
                   experiment_id="3.C.1 Epochs=5000 (欠收敛)")

    # 3.C.2 训练轮数充足 (Epochs=1M) - 基线 (同 1.3)
    # 不重复运行，直接参考 1.3 结果