"""tests/test_predict.py — models/predict.py 单元测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from models.predict import RULPredictor, detect_anomaly


def make_sensor_df(
    temperature=45.0,
    vibration=2.5,
    current=15.0,
    rpm=3000.0,
    rul=1.0,
    fault_type="normal",
    n_rows=70,
):
    """构造测试用传感器 DataFrame。"""
    timestamps = pd.date_range("2026-05-01", periods=n_rows, freq="1min")
    return pd.DataFrame({
        "device_id": ["CNC_001"] * n_rows,
        "device_name": ["数控机床"] * n_rows,
        "timestamp": timestamps,
        "temperature": [temperature] * n_rows,
        "vibration": [vibration] * n_rows,
        "current": [current] * n_rows,
        "rpm": [rpm] * n_rows,
        "fault_type": [fault_type] * n_rows,
        "rul": [rul] * n_rows,
    })


class TestDetectAnomaly:
    """detect_anomaly 边界测试。"""

    def test_normal_returns_no_outliers(self):
        df = make_sensor_df(temperature=45, vibration=2.5, current=15, rpm=3000)
        result = detect_anomaly(df)
        assert result["fault_type"] == "normal"
        assert result["outlier_sensors"] == []

    def test_short_data_returns_normal(self):
        df = make_sensor_df(n_rows=10)
        result = detect_anomaly(df)
        assert result["fault_type"] == "normal"
        assert result["outlier_sensors"] == []

    def test_temperature_outlier_detected(self):
        # 用随机值让sigma>0，最后一行温度跳变（触发3σ检测）
        import numpy as np
        rng = np.random.RandomState(42)
        base_temps = list(rng.randn(69) * 1.5 + 45.0)  # 69个基线随机值
        all_temps = base_temps + [80.0]  # 第70个值=80（触发3σ检测）
        df = make_sensor_df()
        df["temperature"] = all_temps
        df["rul"] = [1.0] * 70
        result = detect_anomaly(df)
        assert "temperature" in result["outlier_sensors"]

    def test_vibration_outlier_detected(self):
        # 用随机值让sigma>0，最后一行振动跳变（触发3σ检测）
        import numpy as np
        rng = np.random.RandomState(42)
        base_vibs = list(rng.randn(69) * 0.3 + 2.5)
        all_vibs = base_vibs + [10.0]
        df = make_sensor_df()
        df["vibration"] = all_vibs
        df["rul"] = [1.0] * 70
        result = detect_anomaly(df)
        assert "vibration" in result["outlier_sensors"]

    def test_fault_type_from_data(self):
        df = make_sensor_df(fault_type="bearing_wear")
        result = detect_anomaly(df)
        assert result["fault_type"] == "bearing_wear"


class TestRULPredictor:
    """RULPredictor 边界测试。"""

    def test_short_data_returns_default(self):
        predictor = RULPredictor()
        df = make_sensor_df(n_rows=30)
        # 模型不存在时返回 1.0
        result = predictor.predict(df)
        assert result == 1.0

    def test_no_model_returns_default(self):
        predictor = RULPredictor(model_path="/nonexistent/model.json")
        df = make_sensor_df()
        assert predictor.predict(df) == 1.0

    def test_model_predicts_in_range(self):
        # 模型存在时会加载，使用模拟数据测试预测范围
        predictor = RULPredictor()
        df = make_sensor_df(n_rows=70)
        # 预测应在 [0, 1] 范围内
        result = predictor.predict(df)
        assert 0.0 <= result <= 1.0
