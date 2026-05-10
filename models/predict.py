"""预测模型 - RUL预测和异常检测（使用真实XGBoost模型）"""

import os
import pandas as pd
import numpy as np


class RULPredictor:
    """RUL预测器 — 加载XGBoost模型进行真实预测。"""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "outputs", "xgboost_rul_model.json"
            )
        self.model_path = model_path
        self._model = None
        self._window_size = 60

    @property
    def model(self):
        """惰性加载XGBoost模型。"""
        if self._model is None:
            from xgboost import XGBRegressor
            self._model = XGBRegressor()
            if os.path.exists(self.model_path):
                self._model.load_model(self.model_path)
            else:
                self._model = None
        return self._model

    def predict(self, df_recent: pd.DataFrame) -> float:
        """对设备数据进行RUL预测。

        Args:
            df_recent: 单台设备的时序数据（至少60行）

        Returns:
            预测RUL值 [0, 1]，模型不存在时返回1.0（默认正常）
        """
        if self.model is None:
            # 模型不存在，返回保守值
            return 1.0

        if len(df_recent) < self._window_size:
            return 1.0

        # 提取窗口特征
        window = df_recent.tail(self._window_size)
        features = self._extract_window_features(window)

        # 模型预测（返回单值或数组）
        pred = self.model.predict(features.reshape(1, -1))
        return float(np.clip(pred[0], 0.0, 1.0))

    def _extract_window_features(self, window: pd.DataFrame) -> np.ndarray:
        """从滚动窗口提取统计特征（与train.py保持一致）。

        Args:
            window: 60行传感器数据

        Returns:
            20维特征向量 [temperature_mean, temperature_std, ..., rpm_trend]
        """
        sensor_cols = ["temperature", "vibration", "current", "rpm"]
        feature_values = []

        for col in sensor_cols:
            vals = window[col].values
            feature_values.extend([
                float(np.mean(vals)),
                float(np.std(vals)),
                float(np.min(vals)),
                float(np.max(vals)),
                float(np.polyfit(np.arange(len(vals)), vals, 1)[0]),
            ])

        return np.array(feature_values, dtype=np.float32)


def detect_anomaly(df_recent: pd.DataFrame) -> dict:
    """检测传感器异常（3σ统计阈值）。

    Args:
        df_recent: 单台设备的时序数据（至少60行）

    Returns:
        {"fault_type": str, "outlier_sensors": list}
    """
    if len(df_recent) < 60:
        return {"fault_type": "normal", "outlier_sensors": []}

    last_row = df_recent.iloc[-1]
    fault_type = last_row.get("fault_type", "normal")

    outlier_sensors = []
    sensor_cols = ["temperature", "vibration", "current", "rpm"]

    for col in sensor_cols:
        recent = df_recent[col].tail(60)
        mu = float(recent.mean())
        sigma = float(recent.std())
        if sigma > 0 and abs(float(last_row[col]) - mu) > 3 * sigma:
            outlier_sensors.append(col)

    return {
        "fault_type": fault_type,
        "outlier_sensors": outlier_sensors
    }


if __name__ == "__main__":
    from data.sensor_simulator import generate_sensor_data

    df = generate_sensor_data()
    predictor = RULPredictor()

    for dev_id in df["device_id"].unique():
        dev_df = df[df["device_id"] == dev_id]
        rul_pred = predictor.predict(dev_df)
        anomaly = detect_anomaly(dev_df)
        name = dev_df.iloc[0]["device_name"]
        print(f"{name} ({dev_id}): RUL={rul_pred:.4f} | 异常: {anomaly['outlier_sensors'] or '无'}")
