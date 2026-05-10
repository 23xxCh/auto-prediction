"""共享特征提取模块 — 统一传感器时序数据的滚动窗口特征计算。

所有需要从60步传感器窗口提取统计特征的地方，统一调用本模块。
避免在多处重复实现相同的特征计算逻辑（特别是polyfit趋势斜率）。
"""

import numpy as np


SENSOR_COLS = ["temperature", "vibration", "current", "rpm"]


def compute_trend(vals: np.ndarray) -> float:
    """计算线性趋势斜率（最小二乘法）。

    Args:
        vals: 一维数组，长度为窗口大小

    Returns:
        斜率值（单位：单位变量/步）
    """
    return float(np.polyfit(np.arange(len(vals)), vals, 1)[0])


def extract_window_features(vals: np.ndarray) -> np.ndarray:
    """从单个传感器的60步窗口提取5维统计特征。

    Args:
        vals: 60-element 传感器值数组

    Returns:
        5维特征向量 [mean, std, min, max, trend_slope]
    """
    return np.array([
        float(np.mean(vals)),
        float(np.std(vals)),
        float(np.min(vals)),
        float(np.max(vals)),
        compute_trend(vals),
    ], dtype=np.float32)


def extract_all_window_features(df_window: np.ndarray, sensor_cols: list = None) -> np.ndarray:
    """从多个传感器列提取完整特征矩阵。

    Args:
        df_window: shape (60, n_sensors) 的传感器窗口数据
        sensor_cols: 传感器列名列表（仅用于注释，不影响计算）

    Returns:
        展平的特征向量，长度 = n_sensors * 5
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_COLS

    n_sensors = min(df_window.shape[1], len(sensor_cols))
    features = []
    for i in range(n_sensors):
        features.extend(extract_window_features(df_window[:, i]))
    return np.array(features, dtype=np.float32)