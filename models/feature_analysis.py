"""特征分析模块 — 从物理意义角度解释传感器信号对故障预测的贡献。"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SENSOR_LABELS_CN = {
    "temperature": "温度",
    "vibration": "振动",
    "current": "电流",
    "rpm": "转速",
}


def analyze_sensor_correlation(df: pd.DataFrame, save_path: str):
    """分析各传感器与RUL的皮尔逊相关系数。

    Args:
        df: 传感器数据集
        save_path: 图片保存路径
    """
    sensor_cols = ["temperature", "vibration", "current", "rpm"]
    correlations = {}
    for col in sensor_cols:
        corr = df[col].corr(df["rul"])
        correlations[SENSOR_LABELS_CN.get(col, col)] = corr

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(correlations.keys(), correlations.values(),
                  color=["#E74C3C" if v < 0 else "#2ECC71" for v in correlations.values()])
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("皮尔逊相关系数", fontsize=11)
    ax.set_title("传感器与RUL相关性分析（负值=传感器值越高，寿命越低）", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, (k, v) in zip(bars, correlations.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_feature_insights(feature_names: list, importance_values: np.ndarray) -> str:
    """基于特征重要性生成可读的物理释义。

    Args:
        feature_names: 特征名列表
        importance_values: 对应重要性值

    Returns:
        可读的中文解释文本
    """
    sorted_indices = np.argsort(importance_values)[::-1]
    top5 = [(feature_names[i], importance_values[i]) for i in sorted_indices[:5]]

    insights = "=== Top 5 关键信号解读 ===\n"
    for name, imp in top5:
        parts = name.split("_")
        sensor = parts[0]
        stat = "_".join(parts[1:])
        stat_map = {"mean": "均值", "std": "标准差", "min": "最小值", "max": "最大值", "trend": "趋势斜率"}
        stat_cn = stat_map.get(stat, stat)
        sensor_cn = SENSOR_LABELS_CN.get(sensor, sensor)
        insights += f"• {sensor_cn}的{stat_cn}（{name}）：重要性 = {imp:.4f}\n"

    insights += "\n=== 物理意义解读 ===\n"
    insights += "1. 振动信号的统计量（尤其是标准差和趋势斜率）是轴承磨损和不平衡故障的最直接指标\n"
    insights += "2. 温度信号的趋势斜率是过热故障的前兆信号\n"
    insights += "3. 转速和电流的联动变化反映了皮带松动等传动机构故障\n"
    insights += "4. 建议实际部署时优先保障振动和温度传感器的数据质量\n"
    return insights


if __name__ == "__main__":
    import os
    from data.sensor_simulator import generate_sensor_data

    os.makedirs("outputs", exist_ok=True)
    df = generate_sensor_data()
    analyze_sensor_correlation(df, "outputs/sensor_correlation.png")
    print("已保存: outputs/sensor_correlation.png")

    # 读取特征重要性
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model("outputs/xgboost_rul_model.json")
    insights = generate_feature_insights(
        [f"feature_{i}" for i in range(len(model.feature_importances_))],
        model.feature_importances_
    )
    print(insights)
