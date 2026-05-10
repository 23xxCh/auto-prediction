"""预测模型 - RUL预测和异常检测（占位实现）"""

import pandas as pd
import numpy as np


class RULPredictor:
    """RUL预测器占位类"""

    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, df_recent):
        """预测RUL百分比 - 返回模拟值"""
        # 取最后一条数据的rul值作为预测结果
        if len(df_recent) > 0:
            return df_recent["rul"].iloc[-1] if "rul" in df_recent.columns else 0.8
        return 0.8


def detect_anomaly(df_recent):
    """检测异常类型 - 返回异常信息字典"""
    if len(df_recent) < 10:
        return {"fault_type": "normal", "outlier_sensors": []}

    last_row = df_recent.iloc[-1]
    fault_type = last_row.get("fault_type", "normal")

    outlier_sensors = []
    # 简单异常检测：温度>50, 振动>6, 电流>20, rpm偏差>20%
    if last_row.get("temperature", 0) > 50:
        outlier_sensors.append("temperature")
    if last_row.get("vibration", 0) > 6:
        outlier_sensors.append("vibration")
    if last_row.get("current", 0) > 20:
        outlier_sensors.append("current")

    return {
        "fault_type": fault_type,
        "outlier_sensors": outlier_sensors
    }