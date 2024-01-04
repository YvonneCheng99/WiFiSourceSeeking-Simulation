import numpy as np
from scipy.stats import multivariate_normal


def multivariate_gaussian_pdf(y, mu, cov):
    """
    计算多元高斯分布的概率密度函数值

    参数：
    - y: 输出向量
    - mu: 均值向量
    - cov: 协方差矩阵
    """
    dim = len(mu)

    # 计算多元高斯分布的概率密度函数
    mvn = multivariate_normal(mean=mu, cov=cov)
    pdf_value = mvn.pdf(y)

    return pdf_value


# 示例
mu = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 2]])

y = np.array([1, 2])

pdf_value = multivariate_gaussian_pdf(y, mu, cov)
print(f"多元高斯分布在 {y} 处的概率密度函数值: {pdf_value}")