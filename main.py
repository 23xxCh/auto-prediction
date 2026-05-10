"""智能装备健康管理系统 — 统一入口脚本。

提供命令行接口，支持：
- 数据生成（传感器模拟 + 故障注入）
- 模型训练（XGBoost RUL预测）
- 看板服务启动（FastAPI + 数字孪生界面）
- 一键全流程执行

用法：
    python main.py --generate-data    # 生成传感器数据
    python main.py --train            # 训练预测模型
    python main.py --dashboard        # 启动看板服务
    python main.py --all              # 一键执行全流程
"""

import argparse
import os
import sys

# 项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="智能装备健康管理系统（IEHM）— 工业设备预测性维护与智能诊断平台"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="生成传感器数据集（7天×3台设备×4类传感器时序数据）"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="训练XGBoost预测模型，生成RUL预测器"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="启动数字孪生看板服务（FastAPI + HMI界面）"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="一键执行全流程：生成数据 → 训练模型 → 启动看板"
    )
    return parser.parse_args()


def step_generate_data():
    """步骤1：生成传感器数据集。

    从 data.sensor_simulator 导入 generate_sensor_data()，
    从 data.fault_injection 导入分析函数，
    生成数据后保存至 outputs/sensor_data.csv，
    并输出故障分布统计、多维时序图、传感器热图。
    """
    print("\n" + "=" * 60)
    print("步骤 1/3：生成传感器数据集")
    print("=" * 60)

    # 导入模块
    from data.sensor_simulator import generate_sensor_data
    from data.fault_injection import analyze_fault_distribution, plot_multidimensional_timeseries, plot_sensor_heatmap

    # 确保输出目录存在
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 生成传感器数据
    print("\n[1/4] 正在生成传感器数据（7天×3台设备×4类传感器）...")
    df = generate_sensor_data()
    print(f"      数据生成完成：{len(df)} 行，{df['device_id'].nunique()} 台设备")

    # 保存数据
    data_path = os.path.join(OUTPUTS_DIR, "sensor_data.csv")
    df.to_csv(data_path, index=False)
    print(f"      数据已保存：{data_path}")

    # 分析故障分布
    print("\n[2/4] 正在分析故障分布...")
    stats = analyze_fault_distribution(df)
    print("\n故障分布统计：")
    print(stats.to_string(index=False))

    # 生成多维时序图
    print("\n[3/4] 正在生成多维时序图（每台设备）...")
    for device_id in df["device_id"].unique():
        output_path = os.path.join(OUTPUTS_DIR, f"timeseries_{device_id}.png")
        plot_multidimensional_timeseries(df, device_id, output_path)

    # 生成传感器热力图
    print("\n[4/4] 正在生成传感器热力图...")
    heatmap_path = os.path.join(OUTPUTS_DIR, "sensor_heatmap.png")
    plot_sensor_heatmap(df, heatmap_path)

    print("\n✓ 数据生成步骤完成")
    return df


def step_train_model():
    """步骤2：训练XGBoost RUL预测模型。

    检查数据是否存在，如不存在则先调用 step_generate_data()。
    从 models.train 导入特征提取和模型训练函数，
    保存模型至 outputs/xgboost_rul_model.json，
    生成特征重要性图和预测散点图。
    """
    print("\n" + "=" * 60)
    print("步骤 2/3：训练XGBoost RUL预测模型")
    print("=" * 60)

    # 导入模块
    from models.train import extract_features, train_xgboost_model, plot_feature_importance, plot_prediction_scatter

    # 检查数据是否存在
    data_path = os.path.join(OUTPUTS_DIR, "sensor_data.csv")
    if not os.path.exists(data_path):
        print("  数据文件不存在，将先生成数据...")
        step_generate_data()

    # 加载数据
    import pandas as pd
    print("\n[1/5] 正在加载传感器数据...")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    print(f"      数据加载完成：{len(df)} 行")

    # 提取特征
    print("\n[2/5] 正在提取时序滚动窗口特征（窗口大小=60）...")
    X, y, feature_names = extract_features(df)
    print(f"      特征矩阵形状：{X.shape}")
    print(f"      特征名称：{feature_names}")

    # 训练模型
    print("\n[3/5] 正在训练XGBoost模型（200棵树，深度6，学习率0.05）...")
    model, metrics, y_test, y_pred = train_xgboost_model(X, y)

    # 保存模型
    print("\n[4/5] 正在保存模型...")
    model_path = os.path.join(OUTPUTS_DIR, "xgboost_rul_model.json")
    model.save_model(model_path)
    print(f"      模型已保存：{model_path}")

    # 生成可视化图表
    print("\n[5/5] 正在生成可视化图表...")
    plot_feature_importance(model, feature_names, os.path.join(OUTPUTS_DIR, "feature_importance.png"))
    plot_prediction_scatter(y_test, y_pred, os.path.join(OUTPUTS_DIR, "prediction_scatter.png"))

    # 打印最终指标
    print("\n" + "-" * 40)
    print("模型训练完成！最终评估指标：")
    print(f"  测试集 R²：   {metrics['test_r2']:.4f}")
    print(f"  测试集 RMSE： {metrics['test_rmse']:.4f}")
    print(f"  测试集 MAE：  {metrics['test_mae']:.4f}")
    print(f"  CV R² 均值：  {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print("-" * 40)
    print("\n✓ 模型训练步骤完成")


def step_start_dashboard():
    """步骤3：启动数字孪生看板服务。

    使用 uvicorn 启动 FastAPI 应用，
    监听 0.0.0.0:8000，
    提供设备状态、遥测数据、RUL趋势、告警、AI诊断报告等REST API。
    """
    print("\n" + "=" * 60)
    print("步骤 3/3：启动数字孪生看板服务")
    print("=" * 60)

    try:
        import uvicorn
        from dashboard import app as dashboard_app
    except ImportError as e:
        print(f"\n✗ 导入看板模块失败：{e}")
        print("  请确保 dashboard 模块已正确配置（需要 dashboard/app.py）")
        sys.exit(1)

    print("\n  正在启动 FastAPI 服务...")
    print("  访问地址：http://localhost:8000")
    print("  按 Ctrl+C 停止服务\n")

    uvicorn.run(
        dashboard_app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )


def main():
    """主入口函数。

    根据命令行参数执行相应步骤：
    - --all：依次执行数据生成、模型训练、看板启动
    - 其他：按需执行单个或多个步骤
    - 无参数：打印帮助信息
    """
    args = parse_args()

    # 一键全流程
    if args.all:
        print("\n" + "█" * 60)
        print("  智能装备健康管理系统 — 一键执行全流程")
        print("█" * 60)
        step_generate_data()
        step_train_model()
        step_start_dashboard()
        return

    # 单步执行
    has_operation = False
    if args.generate_data:
        has_operation = True
        step_generate_data()
    if args.train:
        has_operation = True
        step_train_model()
    if args.dashboard:
        has_operation = True
        step_start_dashboard()

    # 无有效参数
    if not has_operation:
        print("\n请指定操作参数，使用 --help 查看可用选项")
        print("\n示例用法：")
        print("  python main.py --generate-data  # 生成传感器数据")
        print("  python main.py --train         # 训练预测模型")
        print("  python main.py --dashboard     # 启动看板服务")
        print("  python main.py --all            # 一键执行全流程")
        sys.exit(1)

    print("\n✓ 所有指定步骤执行完成")


if __name__ == "__main__":
    main()
