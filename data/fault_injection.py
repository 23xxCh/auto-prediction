"""故障注入引擎与数据可视化模块。

提供故障分布分析、多维时序可视化、传感器热图等功能。
"""

import matplotlib
matplotlib.use("Agg")  # 无头环境使用非交互后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def analyze_fault_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """分析故障分布统计。

    按设备ID、设备名称、故障类型分组统计：
    - 样本数量
    - 平均温度
    - 平均振动
    - 最小剩余使用寿命(RUL)

    Args:
        df: 传感器数据DataFrame，包含device_id, device_name, fault_type,
           temperature, vibration, rul等列

    Returns:
        包含统计指标的DataFrame，按device_id, device_name, fault_type分组
    """
    # 按设备ID、设备名称、故障类型分组统计
    stats = df.groupby(["device_id", "device_name", "fault_type"]).agg(
        count=("temperature", "count"),           # 样本数量
        avg_temp=("temperature", "mean"),          # 平均温度
        avg_vib=("vibration", "mean"),             # 平均振动
        min_rul=("rul", "min")                     # 最小RUL
    ).reset_index()

    # 保留4位小数使输出更整洁
    stats["avg_temp"] = stats["avg_temp"].round(4)
    stats["avg_vib"] = stats["avg_vib"].round(4)
    stats["min_rul"] = stats["min_rul"].round(4)

    return stats


def plot_multidimensional_timeseries(df: pd.DataFrame, device_id: str, save_path: str) -> None:
    """绘制设备多维传感器时序图。

    生成4行子图：温度、振动、电流、转速。
    仅使用该设备最近2天数据，背景标注故障区域。

    Args:
        df: 传感器数据DataFrame
        device_id: 要可视化的设备ID
        save_path: 图片保存路径
    """
    # 筛选设备数据
    device_df = df[df["device_id"] == device_id].copy()
    if device_df.empty:
        raise ValueError(f"未找到设备ID: {device_id}")

    # 转换为datetime类型（如果尚未转换）
    if not pd.api.types.is_datetime64_any_dtype(device_df["timestamp"]):
        device_df["timestamp"] = pd.to_datetime(device_df["timestamp"])

    # 获取最近2天数据（约2880分钟）
    latest_time = device_df["timestamp"].max()
    two_days_ago = latest_time - pd.Timedelta(days=2)
    recent_df = device_df[device_df["timestamp"] >= two_days_ago].copy()

    if recent_df.empty:
        raise ValueError(f"设备 {device_id} 最近2天无数据")

    # 创建4行子图
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{device_id} 多维传感器时序分析（最近2天）", fontsize=14, fontweight="bold")

    # 定义各子图配置
    sensor_config = [
        ("temperature", "温度 (°C)", "Oranges"),      # 温度使用橙色系
        ("vibration", "振动 (mm/s)", "Reds"),         # 振动使用红色系
        ("current", "电流 (A)", "YlOrBr"),            # 电流使用黄棕系
        ("rpm", "转速 (RPM)", "YlOrRd"),              # 转速使用黄红系
    ]

    # 标记故障区域的背景色
    fault_mask = recent_df["fault_type"] != "normal"

    for idx, (col, ylabel, cmap) in enumerate(sensor_config):
        ax = axes[idx]

        # 绘制正常数据线
        ax.plot(recent_df["timestamp"], recent_df[col],
                color=plt.cm.get_cmap(cmap)(0.7), linewidth=1.0, label=col)

        # 标记故障区域背景（红色半透明）
        if fault_mask.any():
            fault_start = None
            in_fault = False
            for i, (is_fault, ts) in enumerate(zip(fault_mask.values, recent_df["timestamp"].values)):
                if is_fault and not in_fault:
                    fault_start = ts
                    in_fault = True
                elif not is_fault and in_fault:
                    ax.axvspan(fault_start, ts, alpha=0.3, color="red", label="故障区域")
                    in_fault = False
            # 处理末尾的故障区域
            if in_fault:
                ax.axvspan(fault_start, recent_df["timestamp"].iloc[-1],
                          alpha=0.3, color="red", label="故障区域")

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

        # Y轴使用YlOrRd样式（通过colormap实现）
        ax.set_facecolor(plt.cm.YlOrRd(0.1))

    # 设置X轴
    axes[-1].set_xlabel("时间", fontsize=10)
    axes[-1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"时序图已保存: {save_path}")


def plot_sensor_heatmap(df: pd.DataFrame, save_path: str) -> None:
    """绘制传感器热力图。

    按设备名称分组计算各传感器均值，进行min-max归一化后
    使用Seaborn热力图可视化，并标注实际均值。

    Args:
        df: 传感器数据DataFrame
        save_path: 图片保存路径
    """
    # 按设备名称分组计算各传感器均值
    sensor_cols = ["temperature", "vibration", "current", "rpm"]
    heatmap_data = df.groupby("device_name")[sensor_cols].mean()

    # 保存原始均值用于标注
    original_means = heatmap_data.copy()

    # Min-Max归一化（每列独立）
    for col in sensor_cols:
        col_min = heatmap_data[col].min()
        col_max = heatmap_data[col].max()
        if col_max - col_min > 0:
            heatmap_data[col] = (heatmap_data[col] - col_min) / (col_max - col_min)
        else:
            heatmap_data[col] = 0

    # 创建热力图
    plt.figure(figsize=(10, 6))

    # 绘制归一化热力图，标注实际均值
    ax = sns.heatmap(
        heatmap_data,
        annot=original_means.round(2),  # 显示原始均值
        fmt="g",
        cmap="YlOrRd",                  # 黄-橙-红配色
        linewidths=0.5,
        cbar_kws={"label": "归一化强度"},
        ax=None
    )

    # 设置中文标题和标签
    plt.title("设备传感器均值热力图", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("传感器类型", fontsize=11)
    plt.ylabel("设备名称", fontsize=11)

    # 设置Y轴刻度标签（设备名称）
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # 调整X轴刻度标签为中文
    sensor_labels = ["温度", "振动", "电流", "转速"]
    ax.set_xticklabels(sensor_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"热力图已保存: {save_path}")


if __name__ == "__main__":
    # 导入传感器数据生成器
    from data.sensor_simulator import generate_sensor_data

    # 生成测试数据
    print("正在生成传感器数据...")
    df = generate_sensor_data()
    print(f"数据生成完成: {len(df)} 行, {df['device_id'].nunique()} 台设备")

    # 分析故障分布
    print("\n=== 故障分布统计 ===")
    stats = analyze_fault_distribution(df)
    print(stats.to_string(index=False))

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 为每个设备生成时序图
    print("\n=== 生成多维时序图 ===")
    for device_id in df["device_id"].unique():
        output_path = os.path.join(output_dir, f"timeseries_{device_id}.png")
        plot_multidimensional_timeseries(df, device_id, output_path)

    # 生成传感器热力图
    print("\n=== 生成传感器热力图 ===")
    heatmap_path = os.path.join(output_dir, "sensor_heatmap.png")
    plot_sensor_heatmap(df, heatmap_path)

    print("\n=== 所有可视化任务完成 ===")
