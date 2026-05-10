"""FastAPI 后端服务 - 设备预测性维护Dashboard API"""

import sys, os
import logging, functools, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncIterator
import pandas as pd
import numpy as np


class DiagnoseRequest(BaseModel):
    """诊断请求模型（预留）"""
    pass


def init_app():
    """应用初始化 - 加载数据和模型

    加载传感器数据（优先从CSV读取，否则生成模拟数据），
    加载XGBoost RUL预测模型（如存在）。
    加载 SHAP TreeExplainer（如存在）。
    加载 LSTM-AE 异常检测器（如存在）。
    """
    global df_global, predictor, shap_explainer, anomaly_detector, kb_retriever

    print("=" * 50)
    print("正在初始化 FastAPI 应用...")

    # 1. 加载传感器数据
    csv_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "sensor_data.csv")
    if os.path.exists(csv_path):
        try:
            df_global = pd.read_csv(csv_path, parse_dates=["timestamp"])
            print(f"  [OK] 已从 {csv_path} 加载传感器数据: {len(df_global)} 行")
        except Exception as e:
            print(f"  [WARN] CSV加载失败 ({e})，改用模拟数据")
            df_global = None
    else:
        print(f"  [INFO] 未找到 {csv_path}，使用模拟数据")

    if df_global is None:
        from data.sensor_simulator import generate_sensor_data
        df_global = generate_sensor_data()
        print(f"  [OK] 已生成模拟传感器数据: {len(df_global)} 行")

    # 2. 加载RUL预测模型
    model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "xgboost_rul_model.json")
    if os.path.exists(model_path):
        try:
            import xgboost as xgb
            # 尝试加载模型（如果失败则使用占位符）
            model = xgb.XGBRegressor()
            model.load_model(model_path)

            # 包装为带predict方法的对象
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model

                def predict(self, df_recent: pd.DataFrame) -> float:
                    """调用模型预测RUL（单行）。
                    特征必须与 models/train.py 的 extract_features() 输出格式一致。
                    """
                    if len(df_recent) < 60:
                        return 1.0
                    from models.features import extract_window_features

                    sensor_cols = ["temperature", "vibration", "current", "rpm"]
                    feature_values = []
                    for col in sensor_cols:
                        vals = df_recent[col].values[-60:]
                        feature_values.extend(extract_window_features(vals))
                    features = np.array(feature_values, dtype=np.float32).reshape(1, -1)
                    pred = self.model.predict(features)
                    return float(np.clip(pred[0], 0.0, 1.0))

            predictor = ModelWrapper(model)
            print(f"  [OK] 已加载XGBoost模型: {model_path}")
        except Exception as e:
            print(f"  [WARN] 模型加载失败 ({e})，使用占位预测器")
            from models.predict import RULPredictor
            predictor = RULPredictor(model_path)
    else:
        print(f"  [INFO] 未找到模型文件 {model_path}，使用占位预测器")
        from models.predict import RULPredictor
        predictor = RULPredictor(model_path)

    print(f"  [OK] 设备数量: {df_global['device_id'].nunique()}")

    # 3. 加载 SHAP TreeExplainer
    try:
        from models.shap import build_shap_explainer
        shap_explainer = build_shap_explainer(model_path)
        if shap_explainer.is_available:
            print(f"  [OK] SHAP TreeExplainer 已加载")
        else:
            print(f"  [INFO] SHAP 不可用（模型未训练），使用降级模式")
            shap_explainer = None
    except Exception as e:
        print(f"  [WARN] SHAP 初始化失败 ({e})，SHAP 不可用")
        shap_explainer = None

    # 4. 加载 LSTM-AE 异常检测器
    try:
        from models.anomaly_lstm import AnomalyLSTM
        anomaly_detector = AnomalyLSTM()
        if anomaly_detector.is_available:
            print(f"  [OK] LSTM-AE 异常检测器已加载")
        else:
            print(f"  [INFO] LSTM-AE 模型未训练，跳过异常检测")
            anomaly_detector = None
    except Exception as e:
        print(f"  [WARN] LSTM-AE 初始化失败 ({e})，异常检测不可用")
        anomaly_detector = None

    # 5. 加载维护知识库（RAG）
    try:
        from rag.retriever import MaintenanceKB
        kb_path = os.path.join(os.path.dirname(__file__), "..", "rag", "maintenance_kb")
        if os.path.isdir(kb_path):
            kb_retriever = MaintenanceKB(kb_path)
            kb_retriever.load()
            print(f"  [OK] 维护知识库已加载：{len(kb_retriever._docs)} 个文档块")
        else:
            print(f"  [INFO] 维护知识库目录不存在，跳过 RAG")
            kb_retriever = None
    except Exception as e:
        print(f"  [WARN] 知识库加载失败 ({e})，RAG 不可用")
        kb_retriever = None

    print("初始化完成" + "=" * 50)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理 - 启动时初始化"""
    init_app()
    yield


# 创建 FastAPI 应用实例
app = FastAPI(
    title="设备预测性维护API",
    description="提供设备状态监控、RUL预测、故障诊断等RESTful接口",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS中间件 - 允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """根路径 - 返回欢迎信息"""
    return {"message": "设备预测性维护Dashboard API", "version": "1.0.0"}


@app.get("/api/devices")
def get_devices():
    """获取所有设备列表及其状态

    Returns:
        List[Dict]: 设备列表，每台设备包含:
            - device_id: 设备ID
            - device_name: 设备名称
            - status: 状态 (normal/warning/danger)
            - rul: 剩余使用寿命百分比 (0-1)
            - rul_percent: RUL百分比字符串
            - temperature: 当前温度
            - vibration: 当前振动
            - current: 当前电流
            - rpm: 当前转速
            - anomaly_sensors: 异常传感器列表
    """
    logger.info("get_devices called")
    if df_global is None:
        raise HTTPException(status_code=500, detail="数据未初始化")

    devices = []
    for device_id in df_global["device_id"].unique():
        df_device = df_global[df_global["device_id"] == device_id].sort_values("timestamp")
        if len(df_device) == 0:
            continue

        last_row = df_device.iloc[-1]
        rul = float(last_row.get("rul", 1.0))

        # 根据RUL判断状态：<0.2危险，<0.5警告，否则正常
        if rul < 0.2:
            status = "danger"
        elif rul < 0.5:
            status = "warning"
        else:
            status = "normal"

        # 统一使用3σ统计异常检测（models.predict.detect_anomaly）
        from models.predict import detect_anomaly
        anomaly = detect_anomaly(df_device)
        outlier_sensors = anomaly["outlier_sensors"]

        devices.append({
            "device_id": device_id,
            "device_name": last_row.get("device_name", device_id),
            "status": status,
            "rul": round(rul, 4),
            "rul_percent": f"{rul * 100:.1f}%",
            "temperature": float(last_row.get("temperature", 0)),
            "vibration": float(last_row.get("vibration", 0)),
            "current": float(last_row.get("current", 0)),
            "rpm": float(last_row.get("rpm", 0)),
            "anomaly_sensors": outlier_sensors
        })

    return devices


@app.get("/api/device/{device_id}/telemetry")
def get_device_telemetry(device_id: str, steps: int = 1440):
    """获取设备遥测数据（时序数据）

    Args:
        device_id: 设备ID
        steps: 返回最近N条数据，默认1440（一天的分钟数）

    Returns:
        Dict: 包含timestamps, temperature, vibration, current, rpm数组
    """
    if df_global is None:
        raise HTTPException(status_code=500, detail="数据未初始化")

    df_device = df_global[df_global["device_id"] == device_id].sort_values("timestamp")
    if len(df_device) == 0:
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 不存在")

    # 取最近steps条数据
    df_recent = df_device.tail(steps).copy()

    return {
        "device_id": device_id,
        "steps": len(df_recent),
        "timestamps": df_recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "temperature": df_recent["temperature"].tolist(),
        "vibration": df_recent["vibration"].tolist(),
        "current": df_recent["current"].tolist(),
        "rpm": df_recent["rpm"].tolist()
    }


@app.get("/api/device/{device_id}/rul")
def get_device_rul(device_id: str, steps: int = 1440):
    """获取设备RUL预测数据

    Args:
        device_id: 设备ID
        steps: 返回最近N条数据，默认1440

    Returns:
        Dict: 包含timestamps, rul_predicted, rul_true数组
    """
    if df_global is None:
        raise HTTPException(status_code=500, detail="数据未初始化")

    df_device = df_global[df_global["device_id"] == device_id].sort_values("timestamp")
    if len(df_device) == 0:
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 不存在")

    # 取最近steps条数据
    df_recent = df_device.tail(steps).copy()

    # 批量预测：一次性提取所有60步窗口特征，调用一次模型
    from models.features import extract_window_features

    sensor_cols = ["temperature", "vibration", "current", "rpm"]
    n = len(df_recent)
    valid_start = max(60, n) - 60  # 第一个有效窗口的起始索引

    if valid_start < n and predictor.model is not None:
        # 通过 predictor.predict() 调用，由 wrapper 内部做特征提取，
        # 保证与训练时的 extract_features() 行为一致（不再直接调用 predictor.model.predict()）。
        rul_predicted_list = [np.nan] * valid_start
        for i in range(valid_start, n):
            df_window = df_recent.iloc[i - 60:i].copy()
            p = predictor.predict(df_window)
            rul_predicted_list.append(round(p, 4))
        rul_predicted_list = rul_predicted_list[:steps]
    else:
        # 模型不可用，返回默认正常值
        rul_predicted_list = [1.0] * steps

    # 真实RUL（从数据中获取）
    rul_true_list = df_recent["rul"].tolist()

    # SHAP 贡献值（如 SHAP 可用）
    shap_values_list = []
    shap_feature_names = None
    if shap_explainer is not None and shap_explainer.is_available and valid_start < n:
        shap_feature_names = shap_explainer.FEATURE_NAMES
        for i in range(valid_start, n):
            df_window = df_recent.iloc[i - 60:i].copy()
            features = predictor._extract_window_features(df_window)
            sv = shap_explainer.explain(features)[0]
            shap_values_list.append([round(float(v), 6) for v in sv])
        # 补齐前面的 nan
        shap_values_list = [[np.nan] * 20] * valid_start + shap_values_list
        shap_values_list = shap_values_list[:steps]
    else:
        shap_values_list = [[np.nan] * 20] * steps

    return {
        "device_id": device_id,
        "steps": len(df_recent),
        "timestamps": df_recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "rul_predicted": rul_predicted_list,
        "rul_true": [round(r, 4) for r in rul_true_list],
        "shap_values": shap_values_list,
        "shap_feature_names": shap_feature_names or []
    }


@app.get("/api/alerts")
def get_alerts():
    """获取告警列表

    Returns RUL < 0.5 或存在异常传感器的设备告警信息

    Returns:
        List[Dict]: 告警列表
    """
    if df_global is None:
        raise HTTPException(status_code=500, detail="数据未初始化")

    alerts = []
    for device_id in df_global["device_id"].unique():
        df_device = df_global[df_global["device_id"] == device_id].sort_values("timestamp")
        if len(df_device) == 0:
            continue

        last_row = df_device.iloc[-1]
        rul = float(last_row.get("rul", 1.0))

        # 统一使用3σ统计异常检测
        from models.predict import detect_anomaly
        anomaly = detect_anomaly(df_device)
        anomaly_sensors = anomaly["outlier_sensors"]

        # 告警条件：RUL < 0.5 或存在异常传感器
        if rul < 0.5 or anomaly_sensors:
            if rul < 0.2:
                severity = "danger"
            elif rul < 0.5:
                severity = "warning"
            else:
                severity = "info"

            alerts.append({
                "device_id": device_id,
                "device_name": last_row.get("device_name", device_id),
                "severity": severity,
                "rul": round(rul, 4),
                "rul_percent": f"{rul * 100:.1f}%",
                "anomaly_sensors": anomaly_sensors,
                "fault_type": last_row.get("fault_type", "normal"),
                "timestamp": last_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(last_row["timestamp"]) else None
            })

    return alerts


@app.get("/api/device/{device_id}/anomaly")

@app.get("/api/knowledge/search")
def knowledge_search(q: str = "", top_k: int = 3):
    """维护知识库语义检索

    Args:
        q: 查询文本（如故障描述或设备症状）
        top_k: 返回最相似的 K 个文档片段，默认 3

    Returns:
        list[dict]: 文档片段列表，包含 text, source, fault_type, score
    """
    if not kb_retriever:
        raise HTTPException(status_code=503, detail="知识库未加载")

    if not q:
        return []

    results = kb_retriever.search(q, top_k=top_k)
    return [
        {
            "text": r["text"][:500],  # 限制长度避免响应过大
            "source": r["metadata"]["source"],
            "fault_type": r["metadata"]["fault_type"],
            "score": round(r["score"], 4)
        }
        for r in results
    ]


def get_device_anomaly(device_id: str):
    """获取设备 LSTM-AE 异常检测结果

    Args:
        device_id: 设备ID

    Returns:
        Dict: 包含 anomaly_score, is_anomaly, threshold
    """
    if df_global is None:
        raise HTTPException(status_code=500, detail="数据未初始化")

    df_device = df_global[df_global["device_id"] == device_id].sort_values("timestamp")
    if len(df_device) == 0:
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 不存在")

    if anomaly_detector is None or not anomaly_detector.is_available:
        raise HTTPException(status_code=503, detail="LSTM-AE 异常检测模型未训练")

    df_recent = df_device.tail(60).copy()
    result = anomaly_detector.detect(df_recent)
    return {
        "device_id": device_id,
        "anomaly_score": result["anomaly_score"],
        "is_anomaly": result["is_anomaly"],
        "threshold": result["threshold"]
    }


@app.post("/api/device/{device_id}/diagnose")
async def diagnose_device(device_id: str):
    logger.info(f"diagnose request: device_id={device_id}")
    """触发设备故障诊断

    Args:
        device_id: 设备ID

    Returns:
        Dict: 诊断报告
    """
    if df_global is None:
        raise HTTPException(status_code=500, detail="数据未初始化")

    df_device = df_global[df_global["device_id"] == device_id].sort_values("timestamp")
    if len(df_device) == 0:
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 不存在")

    # 获取最近60条数据用于诊断
    df_recent = df_device.tail(60).copy()

    # 检测异常
    from models.predict import detect_anomaly
    anomaly_info = detect_anomaly(df_recent)

    # 获取设备名称
    device_name = df_device["device_name"].iloc[-1] if "device_name" in df_device.columns else device_id

    # 提取 SHAP Top-3 特征贡献（如 SHAP 可用）
    shap_top_k = None
    if shap_explainer is not None and shap_explainer.is_available:
        try:
            features = predictor._extract_window_features(df_recent)
            shap_top_k = shap_explainer.top_contributions(features, top_k=3)
        except Exception:
            pass


    # RAG 知识检索（如知识库可用）
    rag_context = None
    if kb_retriever is not None:
        try:
            q_parts = [anomaly_info.get("fault_type", "normal")]
            outlier = anomaly_info.get("outlier_sensors", [])
            if outlier:
                q_parts.append("异常传感器: " + ", ".join(outlier))
            rag_query = " ".join(q_parts)
            if rag_query.strip():
                results = kb_retriever.search(rag_query, top_k=2)
                if results:
                    parts = []
                    for r in results:
                        src = r["metadata"]["source"]
                        body = r["text"][:300]
                        parts.append("[来源: " + src + "] " + body)
                    rag_context = " | ".join(parts)
        except Exception:
            pass
    # 执行诊断（LLM API 调用较慢，在线程池中隔离执行）
    from agent.diagnostic_agent import diagnose
    report = await asyncio.to_thread(diagnose, device_name, df_recent, anomaly_info, device_id, shap_top_k, rag_context)
    logger.info(f"diagnosis done: device_id={device_id}, severity={report.get('severity')}")

    # LRU 缓存（容量上限 _DIAGNOSIS_CACHE_MAX）
    if len(diagnosis_reports) >= _DIAGNOSIS_CACHE_MAX:
        oldest = _diagnosis_cache_order.pop(0)
        diagnosis_reports.pop(oldest, None)
    diagnosis_reports[device_id] = report
    if device_id not in _diagnosis_cache_order:
        _diagnosis_cache_order.append(device_id)

    return report


@app.get("/api/device/{device_id}/diagnosis")
async def get_diagnosis(device_id: str):
    """获取设备诊断报告（如存在缓存则返回缓存，否则触发新诊断）"""
    if device_id in diagnosis_reports:
        return diagnosis_reports[device_id]
    return await diagnose_device(device_id)


# 挂载静态文件目录（html=True 启用SPA路由支持）
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    print(f"  [WARN] 静态目录不存在: {static_dir}，API模式运行")


if __name__ == "__main__":
    import uvicorn
    print("启动 FastAPI 服务: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)