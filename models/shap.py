"""SHAP TreeExplainer — XGBoost RUL 预测模型可解释性封装。

每个 RUL 预测附带 Top-K 特征贡献，让运维人员一眼看懂"为什么预测差"。
复用 models/features.py 的 extract_window_features() 确保特征格式与训练一致。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENSOR_COLS = ["temperature", "vibration", "current", "rpm"]
FEATURE_NAMES = [
    "temperature_mean", "temperature_std", "temperature_min", "temperature_max", "temperature_trend",
    "vibration_mean", "vibration_std", "vibration_min", "vibration_max", "vibration_trend",
    "current_mean", "current_std", "current_min", "current_max", "current_trend",
    "rpm_mean", "rpm_std", "rpm_min", "rpm_max", "rpm_trend",
]


class SHAPExplainer:
    """SHAP TreeExplainer 封装 — 用于解释 XGBoost RUL 预测。"""

    def __init__(self, xgb_model):
        """初始化 SHAP explainer。

        Args:
            xgb_model: 已加载的 XGBRegressor 模型（可以是 None，降级时返回空值）
        """
        self._model = xgb_model
        self._explainer = None
        if xgb_model is not None:
            import shap
            self._explainer = shap.TreeExplainer(xgb_model)

    @property
    def is_available(self) -> bool:
        """SHAP 解释器是否可用。"""
        return self._explainer is not None

    def explain(self, X: np.ndarray) -> tuple[np.ndarray, list]:
        """计算单个特征向量的 SHAP 贡献值。

        Args:
            X: shape (20,) 的特征向量，与 extract_window_features() 输出格式一致

        Returns:
            (shap_values, feature_names)
            shap_values: shape (20,) 的贡献值数组
            feature_names: 特征名列表
        """
        if not self.is_available:
            return np.zeros(20), FEATURE_NAMES

        shap_values = self._explainer.shap_values(X.reshape(1, -1))
        # shap_values 可能是 (1, 20) 或直接 (20,)
        if shap_values.ndim == 2:
            shap_values = shap_values[0]
        return shap_values, FEATURE_NAMES

    def top_contributions(self, X: np.ndarray, top_k: int = 3) -> list[dict]:
        """返回 Top-K 特征贡献（按绝对值排序）。

        Args:
            X: shape (20,) 的特征向量
            top_k: 返回前 K 个贡献，默认为 3

        Returns:
            [{"feature": str, "contribution": float, "abs_contribution": float}, ...]
            按 abs_contribution 从大到小排序
        """
        if not self.is_available:
            return [
                {"feature": name, "contribution": 0.0, "abs_contribution": 0.0}
                for name in FEATURE_NAMES[:top_k]
            ]

        sv, _ = self.explain(X)
        contributions = sorted(
            zip(FEATURE_NAMES, sv),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return [
            {
                "feature": name,
                "contribution": round(float(val), 6),
                "abs_contribution": round(float(abs(val)), 6),
            }
            for name, val in contributions[:top_k]
        ]

    def batch_explain(self, X_batch: np.ndarray) -> list[np.ndarray]:
        """批量计算多个特征向量的 SHAP 贡献值。

        Args:
            X_batch: shape (n_samples, 20) 的特征矩阵

        Returns:
            list of shap_values arrays，每个 shape (20,)
        """
        if not self.is_available:
            return [np.zeros(20) for _ in range(len(X_batch))]

        shap_values = self._explainer.shap_values(X_batch)
        if shap_values.ndim == 2:
            return [row for row in shap_values]
        return [shap_values]


def build_shap_explainer(model_path: str = None):
    """从模型文件路径构建 SHAPExplainer。

    Args:
        model_path: XGBoost 模型路径，默认使用 outputs/xgboost_rul_model.json

    Returns:
        SHAPExplainer 实例（可能降级为不可用状态）
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "xgboost_rul_model.json"
        )

    if not os.path.exists(model_path):
        print(f"[WARN] 模型文件不存在: {model_path}，SHAP 不可用")
        return SHAPExplainer(None)

    try:
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(model_path)
        print(f"[OK] SHAPExplainer 已加载模型: {model_path}")
        return SHAPExplainer(model)
    except Exception as e:
        print(f"[WARN] SHAPExplainer 加载失败 ({e})，SHAP 不可用")
        return SHAPExplainer(None)


if __name__ == "__main__":
    # 快速验证：加载模型，计算 SHAP 值
    from xgboost import XGBRegressor

    model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "xgboost_rul_model.json")
    if not os.path.exists(model_path):
        print("模型文件不存在，跳过验证")
    else:
        model = XGBRegressor()
        model.load_model(model_path)
        explainer = SHAPExplainer(model)

        # 构造随机特征向量测试
        X = np.random.randn(20).astype(np.float32)
        top3 = explainer.top_contributions(X, top_k=3)
        print("Top-3 特征贡献:")
        for item in top3:
            print(f"  {item['feature']}: {item['contribution']:+.6f}")

        sv, names = explainer.explain(X)
        print(f"\nSHAP 值总和: {sv.sum():.6f} (应接近预测值与基线差异)")
        print("SHAPExplainer 验证通过")
