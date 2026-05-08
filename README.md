# ==============================================
# 《高级机器学习理论》课程报告 实验代码
# 算法：线性回归、决策树回归、随机森林回归
# 任务：经典回归仿真问题 + 10折交叉验证
# ==============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------- 1. 生成仿真回归数据 ----------------------
np.random.seed(42)  # 固定随机种子，保证结果可复现
n_samples = 1000    # 样本数
n_features = 5      # 特征数

# 生成特征 X
X = np.random.randn(n_samples, n_features)
# 生成真实权重 + 噪声（构造线性关系）
true_coef = np.array([3, 1.5, -2, 0.5, 2.8])
y = X @ true_coef + np.random.randn(n_samples) * 0.5

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------- 2. 定义模型 ----------------------
models = {
    "线性回归": LinearRegression(),
    "决策树回归": DecisionTreeRegressor(max_depth=5, random_state=42),
    "随机森林回归": RandomForestRegressor(n_estimators=100, random_state=42)
}

# ---------------------- 3. 训练 + 评估 + 10折交叉验证 ----------------------
results = []

for name, model in models.items():
    # 训练
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # 10折交叉验证
    cv_rmse = -cross_val_score(model, X, y, cv=10, scoring="neg_root_mean_squared_error").mean()
    
    results.append([name, round(rmse,3), round(r2,3), round(cv_rmse,3)])

# ---------------------- 4. 输出结果表格 ----------------------
df = pd.DataFrame(results, columns=["算法", "RMSE", "R²", "10折CV_RMSE"])
print("="*60)
print("三种机器学习模型在回归仿真问题上的性能对比")
print("="*60)
print(df.to_string(index=False))
print("="*60)
