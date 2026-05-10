"""Prompt模板 — 为LLM诊断Agent准备结构化提示词。"""

import numpy as np

DIAGNOSIS_SYSTEM_PROMPT = """你是一名资深工业装备诊断专家，拥有20年数控机床、纺织设备、工业机器人维修经验。
你的任务是基于传感器数据上下文，对异常设备进行故障诊断分析。请严格按以下格式输出（中文）：
## 故障根因分析
[具体分析]
## 建议操作步骤
1. [步骤1]
2. [步骤2]
...
## 所需备件清单
- [备件1]
- [备件2]
..."""

def build_diagnosis_prompt(device_name, recent_data_str, anomaly_info):
    """构建完整的诊断Prompt

    Args:
        device_name: 设备名称
        recent_data_str: 最近传感器数据的字符串表示
        anomaly_info: 异常信息字典，包含fault_type和outlier_sensors

    Returns:
        格式化的用户诊断提示词
    """
    fault_type = anomaly_info.get("fault_type", "normal")
    outlier_sensors = anomaly_info.get("outlier_sensors", [])

    prompt = f"""## 设备信息
设备名称：{device_name}

## 最近60步传感器数据统计摘要
{recent_data_str}

## 异常信息
故障类型：{fault_type}
异常传感器：{', '.join(outlier_sensors) if outlier_sensors else '无'}

请根据以上信息进行故障诊断分析。
"""
    return prompt


def build_data_context(df_recent):
    """将最近60步传感器数据构建为可读的上下文统计摘要

    Args:
        df_recent: 最近60步的传感器数据DataFrame

    Returns:
        格式化的数据统计摘要字符串
    """
    sensors = ["temperature", "vibration", "current", "rpm"]
    sensor_names = {"temperature": "温度", "vibration": "振动", "current": "电流", "rpm": "转速"}
    sensor_units = {"temperature": "°C", "vibration": "mm/s", "current": "A", "rpm": "RPM"}

    context_lines = []

    for sensor in sensors:
        if sensor not in df_recent.columns:
            continue

        values = df_recent[sensor].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # 计算趋势（简单线性回归斜率）
        if len(values) > 1:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            if slope > 0.01:
                trend = "上升"
            elif slope < -0.01:
                trend = "下降"
            else:
                trend = "稳定"
        else:
            trend = "稳定"

        unit = sensor_units.get(sensor, "")
        context_lines.append(
            f"- {sensor_names.get(sensor, sensor)}({sensor}): 均值={mean_val:.2f}{unit}, "
            f"标准差={std_val:.2f}, 范围=[{min_val:.2f}, {max_val:.2f}], 趋势={trend}"
        )

    return "\n".join(context_lines)