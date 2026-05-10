"""LLM诊断Agent - 基于传感器数据对异常设备进行故障诊断"""

import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime

# 从环境变量获取API配置
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-placeholder")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# 故障类型到诊断报告的映射（后备方案）
FALLBACK_DIAGNOSIS = {
    "bearing_wear": {
        "fault_analysis": "轴承磨损故障：振动信号异常增大，频谱分析显示高频成分增加，轴承表面可能存在磨损、裂纹或润滑不良。建议停机检查轴承状态。",
        "steps": [
            "1. 停机并断开电源，确保设备完全停止",
            "2. 拆卸轴承端盖，检查轴承外观有无明显磨损、裂纹或变色",
            "3. 使用振动分析仪测量轴承座振动频谱，重点关注高频段",
            "4. 检查润滑油量及清洁度，必要时更换润滑油",
            "5. 如轴承间隙超标或表面损伤，立即更换同规格轴承"
        ],
        "parts_needed": [
            "精密轴承（型号参考设备手册）",
            "轴承专用润滑脂",
            "密封圈/O型圈",
            "固定螺栓（部分）"
        ]
    },
    "overheating": {
        "fault_analysis": "过热故障：温度传感器检测到温度持续升高，可能原因包括冷却系统失效、负载过大、通风不良或传感器故障。建议立即降低负载并检查冷却系统。",
        "steps": [
            "1. 立即降低设备负载或停机冷却",
            "2. 检查冷却风扇是否正常运转，清理散热片灰尘",
            "3. 检查冷却液液位和流量，补充或更换冷却液",
            "4. 检查温度传感器接线是否松动，传感器阻值是否正常",
            "5. 如温度仍异常，可能是功率器件老化，需进一步检修"
        ],
        "parts_needed": [
            "冷却风扇组件",
            "温度传感器探头",
            "散热硅脂",
            "冷却液（规格见设备手册）"
        ]
    },
    "belt_loose": {
        "fault_analysis": "皮带松动故障：转速波动异常，电流信号显示负载变化不稳定，皮带可能存在松弛、磨损或张紧力不足问题。",
        "steps": [
            "1. 停机并断开电源",
            "2. 检查皮带张紧度，用拇指按压皮带中间位置，正常下沉量约10-15mm",
            "3. 检查皮带外观有无裂纹、磨损、剥离等损伤",
            "4. 如皮带松弛，调整张紧轮位置或更换张紧弹簧",
            "5. 更换老化或损坏的皮带，并重新张紧"
        ],
        "parts_needed": [
            "传动皮带（规格见设备手册）",
            "张紧弹簧",
            "张紧轮轴承",
            "皮带轮定位垫片"
        ]
    },
    "normal": {
        "fault_analysis": "设备当前运行正常，各项传感器参数在正常范围内，未检测到明显异常特征。",
        "steps": [
            "1. 继续正常监控设备运行状态",
            "2. 定期记录传感器数据趋势",
            "3. 按计划进行预防性维护",
            "4. 如发现参数异常，及时复检",
            "5. 保持设备清洁和良好运行环境"
        ],
        "parts_needed": [
            "无需备件",
            "定期维护用清洁工具",
            "润滑油脂（常规保养）",
            "备件库存建议：常用轴承、皮带、温度传感器各1套"
        ]
    }
}


def call_llm_api(system_prompt, user_message, max_retries=2):
    """调用LLM API进行诊断分析

    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        max_retries: 最大重试次数

    Returns:
        API响应的文本内容

    Raises:
        RuntimeError: 当所有重试都失败时抛出
    """
    url = f"{LLM_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                # 非200状态码，记录错误并重试
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"API调用超时 (尝试 {attempt + 1}/{max_retries})")
        except requests.exceptions.ConnectionError as e:
            print(f"API连接错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

    raise RuntimeError(f"LLM API调用失败，已重试 {max_retries} 次")


def parse_diagnosis_output(text):
    """解析LLM输出的诊断报告

    Args:
        text: LLM返回的原始文本

    Returns:
        dict: 包含 fault_analysis, steps, parts_needed 的字典
    """
    result = {
        "fault_analysis": "",
        "steps": [],
        "parts_needed": []
    }

    current_section = None
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if "故障根因分析" in line:
            current_section = "fault_analysis"
            continue
        elif "建议操作步骤" in line:
            current_section = "steps"
            continue
        elif "所需备件清单" in line:
            current_section = "parts_needed"
            continue

        if current_section == "fault_analysis" and line:
            result["fault_analysis"] += line + " "
        elif current_section == "steps" and line:
            # 步骤可能是 "1. xxx" 或 "xxx" 格式
            step_text = line.lstrip("0123456789.、 ").strip()
            if step_text:
                result["steps"].append(step_text)
        elif current_section == "parts_needed" and line:
            part_text = line.lstrip("-*、 ").strip()
            if part_text:
                result["parts_needed"].append(part_text)

    # 清理fault_analysis末尾空格
    result["fault_analysis"] = result["fault_analysis"].strip()

    return result


def diagnose(device_name, df_recent, anomaly_info, device_id, shap_top_k=None, rag_context=None):
    """执行设备故障诊断

    Args:
        device_name: 设备名称
        df_recent: 最近60步的传感器数据
        anomaly_info: 异常信息字典，包含fault_type和outlier_sensors
        device_id: 设备ID
        shap_top_k: SHAP Top-K 特征贡献列表，每项为 {"feature": str, "contribution": float}

    Returns:
        dict: 诊断报告字典
    """
    from agent.prompt_templates import DIAGNOSIS_SYSTEM_PROMPT, build_diagnosis_prompt, build_data_context

    # 获取当前时间戳
    timestamp = pd.Timestamp.now().isoformat()

    # 根据RUL确定严重程度
    rul = df_recent["rul"].iloc[-1] if "rul" in df_recent.columns and len(df_recent) > 0 else 1.0
    if rul < 0.2:
        severity = "high"
    elif rul < 0.5:
        severity = "medium"
    else:
        severity = "low"

    try:
        # 构建数据上下文
        recent_data_str = build_data_context(df_recent)

        # 构建诊断Prompt（包含 SHAP 特征贡献）
        user_message = build_diagnosis_prompt(device_name, recent_data_str, anomaly_info, shap_top_k, rag_context)

        # 调用LLM API
        response_text = call_llm_api(DIAGNOSIS_SYSTEM_PROMPT, user_message)

        # 解析输出
        parsed = parse_diagnosis_output(response_text)

        return {
            "device_id": device_id,
            "timestamp": timestamp,
            "fault_analysis": parsed["fault_analysis"],
            "steps": parsed["steps"],
            "parts_needed": parsed["parts_needed"],
            "severity": severity,
            "source": "llm"
        }
    except Exception as e:
        # API调用失败，使用后备方案
        print(f"LLM API调用失败，使用后备方案: {str(e)}")
        fault_type = anomaly_info.get("fault_type", "normal")
        fallback = FALLBACK_DIAGNOSIS.get(fault_type, FALLBACK_DIAGNOSIS["normal"])

        return {
            "device_id": device_id,
            "timestamp": timestamp,
            "fault_analysis": fallback["fault_analysis"],
            "steps": fallback["steps"],
            "parts_needed": fallback["parts_needed"],
            "severity": severity,
            "source": "fallback"
        }


def print_report(report):
    """打印诊断报告"""
    print("\n" + "=" * 60)
    print(f"设备ID: {report['device_id']}")
    print(f"时间戳: {report['timestamp']}")
    print(f"严重程度: {report['severity'].upper()}")
    print(f"数据来源: {report['source']}")
    print("-" * 60)
    print(f"故障分析: {report['fault_analysis']}")
    print("-" * 60)
    print("建议操作步骤:")
    for i, step in enumerate(report['steps'], 1):
        print(f"  {i}. {step}")
    print("-" * 60)
    print("所需备件:")
    for part in report['parts_needed']:
        print(f"  - {part}")
    print("=" * 60)


if __name__ == "__main__":
    # 导入所需模块
    from data.sensor_simulator import generate_sensor_data
    from models.predict import RULPredictor, detect_anomaly

    # 生成传感器数据
    print("正在生成传感器数据...")
    df = generate_sensor_data()

    # 获取每台设备的最近60条数据
    devices = df["device_id"].unique()

    # 创建预测器实例
    predictor = RULPredictor()

    print("\n开始诊断分析...")
    print("-" * 40)

    for device_id in devices:
        # 筛选该设备的数据
        df_device = df[df["device_id"] == device_id].sort_values("timestamp")

        # 取最近60条数据
        df_recent = df_device.tail(60).copy()

        # 获取设备名称
        device_name = df_device["device_name"].iloc[0] if len(df_device) > 0 else device_id

        # 预测RUL
        rul_predicted = predictor.predict(df_recent)
        print(f"\n设备 {device_id} ({device_name}) - 预测RUL: {rul_predicted:.2%}")

        # 检测异常
        anomaly_info = detect_anomaly(df_recent)
        print(f"  故障类型: {anomaly_info['fault_type']}")
        print(f"  异常传感器: {anomaly_info.get('outlier_sensors', [])}")

        # 执行诊断
        report = diagnose(device_name, df_recent, anomaly_info, device_id)

        # 打印报告
        print_report(report)