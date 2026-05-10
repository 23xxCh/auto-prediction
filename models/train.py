"""XGBoost RUL预测模型训练与评估模块。"""

import os
import sys

# 将项目根目录添加到Python路径，以便正确导入模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，避免在无GUI环境报错
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.features import compute_trend

# ============ 常量定义 ============
WINDOW_SIZE = 60  # 滚动窗口大小
TEST_SPLIT_RATIO = 0.2  # 测试集比例（时间序列不shuffle）
N_CV_FOLDS = 5  # 交叉验证折数
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0
}


def extract_features(df: pd.DataFrame):
    """提取时序滚动窗口特征。

    对每个传感器列计算滚动窗口(WINDOW_SIZE)内的:
    mean, std, min, max, trend(线性斜率)

    Args:
        df: 传感器数据DataFrame，包含列:
            device_id, timestamp, temperature, vibration, current, rpm, rul

    Returns:
        X: 特征矩阵(numpy array)
        y: 目标变量(numpy array)
        feature_names: 特征名称列表
    """
    sensor_columns = ["temperature", "vibration", "current", "rpm"]
    feature_names = []

    # 为每个传感器列生成滚动特征名称
    for col in sensor_columns:
        for stat in ["mean", "std", "min", "max", "trend"]:
            feature_names.append(f"{col}_{stat}")

    # 创建结果列表
    all_X = []
    all_y = []

    # 按设备分组处理，保持时间序列顺序
    for device_id, group in df.groupby("device_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)

        # 跳过数据量不足WINDOW_SIZE的设备
        if n < WINDOW_SIZE:
            print(f"  [WARN] 设备 {device_id} 数据量不足({n} < {WINDOW_SIZE})，已跳过")
            continue

        # 初始化特征矩阵
        X_device = np.zeros((n - WINDOW_SIZE + 1, len(sensor_columns) * 5))
        y_device = group["rul"].values[WINDOW_SIZE - 1:]

        col_idx = 0
        for col in sensor_columns:
            values = group[col].values

            for i in range(n - WINDOW_SIZE + 1):
                window = values[i:i + WINDOW_SIZE]
                X_device[i, col_idx:col_idx + 5] = compute_trend(window)

            col_idx += 5

        all_X.append(X_device)
        all_y.append(y_device)

    # 合并所有设备数据
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # 删除包含NaN的行（首部WINDOW_SIZE-1行理论上已排除，但以防万一）
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, feature_names


def train_xgboost_model(X: np.ndarray, y: np.ndarray):
    """训练XGBoost回归模型并进行评估。

    使用时间序列分割(80/20)划分训练集和测试集，
    进行5折TimeSeriesSplit交叉验证。

    Args:
        X: 特征矩阵
        y: 目标变量

    Returns:
        model: 训练好的XGBRegressor模型
        metrics: 评估指标字典，包含测试集和交叉验证结果
        y_test: 测试集真实值
        y_pred: 测试集预测值
    """
    # 计算时间序列分割点
    n_samples = len(X)
    split_idx = int(n_samples * (1 - TEST_SPLIT_RATIO))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 训练XGBoost模型
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # 测试集预测与评估
    y_pred = model.predict(X_test)

    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)

    print("\n=== 测试集评估结果 ===")
    print(f"R2  (决定系数): {test_r2:.4f}")
    print(f"RMSE(均方根误差): {test_rmse:.4f}")
    print(f"MAE (平均绝对误差): {test_mae:.4f}")

    # 5折时间序列交叉验证
    print(f"\n=== {N_CV_FOLDS}折TimeSeriesSplit交叉验证 ===")
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)

    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

        cv_model = XGBRegressor(**MODEL_PARAMS)
        cv_model.fit(X_cv_train, y_cv_train)

        y_cv_pred = cv_model.predict(X_cv_val)

        cv_r2 = r2_score(y_cv_val, y_cv_pred)
        cv_rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
        cv_mae = mean_absolute_error(y_cv_val, y_cv_pred)

        cv_r2_scores.append(cv_r2)
        cv_rmse_scores.append(cv_rmse)
        cv_mae_scores.append(cv_mae)

        print(f"Fold {fold_idx}: R2={cv_r2:.4f}, RMSE={cv_rmse:.4f}, MAE={cv_mae:.4f}")

    print(f"\n交叉验证均值+/-标准差:")
    print(f"R2:  {np.mean(cv_r2_scores):.4f} +/- {np.std(cv_r2_scores):.4f}")
    print(f"RMSE: {np.mean(cv_rmse_scores):.4f} +/- {np.std(cv_rmse_scores):.4f}")
    print(f"MAE:  {np.mean(cv_mae_scores):.4f} +/- {np.std(cv_mae_scores):.4f}")

    metrics = {
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "cv_r2_mean": np.mean(cv_r2_scores),
        "cv_r2_std": np.std(cv_r2_scores),
        "cv_rmse_mean": np.mean(cv_rmse_scores),
        "cv_rmse_std": np.std(cv_rmse_scores),
        "cv_mae_mean": np.mean(cv_mae_scores),
        "cv_mae_std": np.std(cv_mae_scores),
    }

    return model, metrics, y_test, y_pred


def plot_feature_importance(model: XGBRegressor, feature_names: list, save_path: str):
    """绘制特征重要性图（Top 15，水平条形图）。

    Args:
        model: 训练好的XGBoost模型
        feature_names: 特征名称列表
        save_path: 图片保存路径
    """
    # 获取特征重要性
    importances = model.feature_importances_

    # 按重要性排序，取Top 15
    indices = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # 绘制水平条形图
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances, color="darkorange")
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("特征重要性 (Feature Importance)")
    plt.ylabel("特征名称")
    plt.title("XGBoost RUL预测 - Top 15 特征重要性")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"特征重要性图已保存: {save_path}")


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """绘制预测散点图（含45度参考线）。

    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 图片保存路径
    """
    plt.figure(figsize=(8, 8))

    # 绘制散点
    plt.scatter(y_true, y_pred, alpha=0.5, color="steelblue", s=20)

    # 绘制45度参考线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="45度参考线")

    plt.xlabel("真实RUL值 (True RUL)")
    plt.ylabel("预测RUL值 (Predicted RUL)")
    plt.title("XGBoost RUL预测 - 预测值 vs 真实值")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"预测散点图已保存: {save_path}")


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)

    # 导入传感器数据生成器
    from data.sensor_simulator import generate_sensor_data

    # 生成传感器数据并保存
    print("正在生成传感器数据...")
    df = generate_sensor_data()
    df.to_csv("outputs/sensor_data.csv", index=False)
    print(f"传感器数据已保存: outputs/sensor_data.csv")

    # 提取特征
    print("\n正在提取特征...")
    X, y, feature_names = extract_features(df)
    print(f"特征矩阵形状: {X.shape}, 目标变量长度: {len(y)}")
    print(f"特征数量: {len(feature_names)}")

    # 训练模型
    print("\n正在训练XGBoost模型...")
    model, metrics, y_test, y_pred = train_xgboost_model(X, y)

    # 保存模型
    model_path = "outputs/xgboost_rul_model.json"
    model.save_model(model_path)
    print(f"\n模型已保存: {model_path}")

    # 绘制并保存图表
    print("\n正在生成可视化图表...")
    plot_feature_importance(model, feature_names, "outputs/feature_importance.png")
    plot_prediction_scatter(y_test, y_pred, "outputs/prediction_scatter.png")

    # 打印最终指标摘要
    print("\n" + "=" * 50)
    print("训练完成！最终评估指标:")
    print(f"测试集 R2:  {metrics['test_r2']:.4f}")
    print(f"测试集 RMSE: {metrics['test_rmse']:.4f}")
    print(f"测试集 MAE:  {metrics['test_mae']:.4f}")
    print(f"CV R2均值:   {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}")
    print("=" * 50)