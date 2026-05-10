"""tests/test_shap_lstm.py — SHAP + LSTM-AE 单元测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


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


class TestSHAPExplainer:
    """SHAPExplainer 降级和核心功能测试。"""

    def test_no_model_returns_zero_shap(self):
        """模型为 None 时，SHAP 返回全零值。"""
        from models.shap import SHAPExplainer
        explainer = SHAPExplainer(None)
        assert not explainer.is_available

        X = np.random.randn(20).astype(np.float32)
        sv, names = explainer.explain(X)
        assert len(sv) == 20
        assert all(v == 0.0 for v in sv)
        assert len(names) == 20

    def test_top_contributions_returns_k_items(self):
        """Top-K 返回正确数量。"""
        from models.shap import SHAPExplainer
        explainer = SHAPExplainer(None)

        X = np.random.randn(20).astype(np.float32)
        top5 = explainer.top_contributions(X, top_k=5)
        assert len(top5) == 5

        top3 = explainer.top_contributions(X, top_k=3)
        assert len(top3) == 3

    def test_feature_names_correct(self):
        """FEATURE_NAMES 包含 4 个传感器 × 5 统计量。"""
        from models.shap import SHAPExplainer, FEATURE_NAMES
        assert len(FEATURE_NAMES) == 20
        # 检查名称包含传感器列名
        assert any("temperature" in n for n in FEATURE_NAMES)
        assert any("vibration" in n for n in FEATURE_NAMES)
        assert any("rpm" in n for n in FEATURE_NAMES)

    def test_batch_explain_returns_correct_count(self):
        """批量解释返回正确数量。"""
        from models.shap import SHAPExplainer
        explainer = SHAPExplainer(None)

        X_batch = np.random.randn(5, 20).astype(np.float32)
        results = explainer.batch_explain(X_batch)
        assert len(results) == 5
        for sv in results:
            assert len(sv) == 20

    def test_build_shap_explainer_nonexistent_path(self):
        """不存在的模型路径返回降级 explainer。"""
        from models.shap import build_shap_explainer
        explainer = build_shap_explainer("/nonexistent/path.json")
        assert not explainer.is_available


class TestAnomalyLSTM:
    """AnomalyLSTM 降级和接口测试。"""

    def test_no_model_returns_zero_score(self):
        """模型不存在时，score 返回 0.0。"""
        from models.anomaly_lstm import AnomalyLSTM
        detector = AnomalyLSTM(model_path="/nonexistent/model.h5")
        assert not detector.is_available

        df = make_sensor_df(n_rows=70)
        score = detector.score(df)
        assert score == 0.0

    def test_detect_returns_dict_structure(self):
        """detect() 返回正确字段结构。"""
        from models.anomaly_lstm import AnomalyLSTM
        detector = AnomalyLSTM(model_path="/nonexistent/model.h5")

        df = make_sensor_df(n_rows=70)
        result = detector.detect(df)

        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert "threshold" in result
        assert isinstance(result["is_anomaly"], bool)

    def test_short_data_raises_error(self):
        """数据不足 60 行时，detect() 抛出 ValueError。"""
        from models.anomaly_lstm import AnomalyLSTM
        detector = AnomalyLSTM(model_path="/nonexistent/model.h5")

        df = make_sensor_df(n_rows=30)
        try:
            detector.detect(df)  # detect() 现在会在前面验证数据量
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "60" in str(e)

    def test_default_threshold(self):
        """默认阈值为 0.05。"""
        from models.anomaly_lstm import AnomalyLSTM, ANOMALY_THRESHOLD
        assert ANOMALY_THRESHOLD == 0.05

        detector = AnomalyLSTM()
        assert detector.threshold == 0.05

    def test_custom_threshold(self):
        """可设置自定义阈值。"""
        from models.anomaly_lstm import AnomalyLSTM
        detector = AnomalyLSTM(threshold=0.1)
        assert detector.threshold == 0.1
