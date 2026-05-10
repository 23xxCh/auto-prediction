"""LSTM AutoEncoder — 无监督时序异常检测。

训练阶段学习正常时序模式的重构误差，推理阶段：
- 重构误差低 → 正常
- 重构误差高 → 异常（可能存在新型故障，XGBoost 盲区）

架构：Encoder-Decoder 对称 LSTM
- Encoder: Input(60,20) → LSTM(64) → LSTM(32) → Latent(8)
- Decoder: Latent(8) → LSTM(32) → LSTM(64) → Output(60,20)
- Loss: MSE
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# TensorFlow 必须在任何 import keras/tensorflow 之前设置
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from tensorflow import keras
    from tensorflow.keras import layers as klayers
except ImportError:
    keras = None


WINDOW_SIZE = 60
LATENT_DIM = 8
LSTM_HIDDEN_1 = 32
LSTM_HIDDEN_2 = 64
ANOMALY_THRESHOLD = 0.05  # 重构误差阈值（可通过训练数据自动确定）


def build_lstm_autoencoder(input_dim: int = 20):
    """构建 LSTM AutoEncoder 模型。

    Args:
        input_dim: 特征维度（默认 20，对应 extract_window_features 输出）

    Returns:
        编译好的 Keras Model
    """
    inputs = keras.Input(shape=(WINDOW_SIZE, input_dim), name="encoder_input")

    # Encoder
    x = klayers.LSTM(LSTM_HIDDEN_2, return_sequences=True, name="enc_lstm1")(inputs)
    x = klayers.LSTM(LATENT_DIM, return_sequences=False, name="enc_lstm2")(x)
    encoded = klayers.RepeatVector(WINDOW_SIZE, name="repeat_vector")(x)

    # Decoder
    x = klayers.LSTM(LSTM_HIDDEN_2, return_sequences=True, name="dec_lstm1")(encoded)
    x = klayers.LSTM(LSTM_HIDDEN_1, return_sequences=True, name="dec_lstm2")(x)
    outputs = klayers.TimeDistributed(klayers.Dense(input_dim), name="decoder_output")(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


class AnomalyLSTM:
    """LSTM AutoEncoder 异常检测器。"""

    def __init__(self, model_path: str = None, threshold: float = None):
        """初始化异常检测器。

        Args:
            model_path: 模型文件路径（.h5）
            threshold: 重构误差异常阈值，默认 0.05
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "outputs", "lstm_anomaly_model.h5"
            )
        self.model_path = model_path
        self.threshold = threshold if threshold is not None else ANOMALY_THRESHOLD
        self._model = None

    @property
    def model(self):
        if self._model is None and os.path.exists(self.model_path):
            try:
                self._model = keras.models.load_model(self.model_path)
            except Exception as e:
                print(f"[WARN] LSTM 模型加载失败 ({e})")
                self._model = None
        return self._model

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def _prepare_window(self, df_window: pd.DataFrame) -> np.ndarray:
        """将 DataFrame 窗口转换为 (60, 20) numpy 数组。

        Args:
            df_window: 至少 60 行传感器数据 DataFrame

        Returns:
            shape (60, 20) 的 numpy 数组
        """
        from models.features import extract_window_features

        if len(df_window) < WINDOW_SIZE:
            raise ValueError(f"数据不足 {WINDOW_SIZE} 行，当前 {len(df_window)} 行")

        window = df_window.tail(WINDOW_SIZE)
        sensor_cols = ["temperature", "vibration", "current", "rpm"]

        # 将 60 行 DataFrame 转换为 60 个 20 维特征向量
        X = np.zeros((WINDOW_SIZE, len(sensor_cols) * 5), dtype=np.float32)
        for i, col in enumerate(sensor_cols):
            vals = window[col].values
            features = extract_window_features(vals)  # shape (5,)
            for step in range(WINDOW_SIZE):
                X[step, i * 5:(i + 1) * 5] = features  # 每一步用相同窗口的特征

        # 更正确的方式：构造 60 个滑动窗口
        X = np.zeros((WINDOW_SIZE, len(sensor_cols) * 5), dtype=np.float32)
        values_dict = {col: window[col].values for col in sensor_cols}
        for step in range(WINDOW_SIZE):
            for i, col in enumerate(sensor_cols):
                start = step
                end = step + WINDOW_SIZE
                if end <= len(values_dict[col]):
                    vals = values_dict[col][start:end]
                else:
                    vals = values_dict[col][-WINDOW_SIZE:]
                feat = extract_window_features(vals)
                X[step, i * 5:(i + 1) * 5] = feat

        return X

    def score(self, df_window: pd.DataFrame) -> float:
        """计算重构误差分数（越高越异常）。

        Args:
            df_window: 至少 60 行传感器数据

        Returns:
            float，重构误差 MSE
        """
        if not self.is_available:
            return 0.0

        X = self._prepare_window(df_window)
        X_recon = self.model.predict(X.reshape(1, WINDOW_SIZE, -1), verbose=0)
        mse = float(np.mean((X - X_recon.reshape(WINDOW_SIZE, -1)) ** 2))
        return mse

    def detect(self, df_window: pd.DataFrame) -> dict:
        """异常检测主入口。

        Args:
            df_window: 至少 60 行传感器数据

        Returns:
            {
                "anomaly_score": float,  # 重构误差
                "is_anomaly": bool,       # 是否超过阈值
                "threshold": float        # 阈值
            }
        """
        # 先验证数据量（与 model 可用性无关）
        if len(df_window) < WINDOW_SIZE:
            raise ValueError(f"数据不足 {WINDOW_SIZE} 行，当前 {len(df_window)} 行")

        score = self.score(df_window)
        return {
            "anomaly_score": round(score, 8),
            "is_anomaly": score > self.threshold,
            "threshold": self.threshold,
        }

    def set_threshold_from_data(self, df_normal: pd.DataFrame, percentile: float = 95.0):
        """从正常数据自动确定阈值。

        Args:
            df_normal: 正常状态下的传感器数据
            percentile: 取重构误差分布的百分位作为阈值，默认 95%
        """
        scores = []
        for device_id in df_normal["device_id"].unique():
            df_dev = df_normal[df_normal["device_id"] == device_id].sort_values("timestamp")
            for i in range(0, len(df_dev) - WINDOW_SIZE + 1, WINDOW_SIZE):
                window = df_dev.iloc[i:i + WINDOW_SIZE]
                scores.append(self.score(window))
        if scores:
            self.threshold = float(np.percentile(scores, percentile))


def train_lstm_autoencoder(
    df: pd.DataFrame,
    save_path: str = None,
    epochs: int = 30,
    batch_size: int = 32,
    val_split: float = 0.1,
) -> AnomalyLSTM:
    """训练 LSTM AutoEncoder 并保存。

    Args:
        df: 传感器数据 DataFrame
        save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批大小
        val_split: 验证集比例

    Returns:
        训练好的 AnomalyLSTM 实例
    """
    if keras is None:
        raise ImportError("TensorFlow 未安装，无法训练 LSTM-AE。请运行: pip install tensorflow")

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "lstm_anomaly_model.h5"
        )

    from models.features import extract_window_features

    sensor_cols = ["temperature", "vibration", "current", "rpm"]

    # 准备训练数据：从滑动窗口生成特征矩阵
    all_windows = []
    for device_id, group in df.groupby("device_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)
        for i in range(0, n - WINDOW_SIZE + 1, WINDOW_SIZE):  # 非重叠窗口
            window = group.iloc[i:i + WINDOW_SIZE]
            X = np.zeros((WINDOW_SIZE, len(sensor_cols) * 5), dtype=np.float32)
            for step in range(WINDOW_SIZE):
                for j, col in enumerate(sensor_cols):
                    vals = window[col].values
                    feat = extract_window_features(vals)
                    X[step, j * 5:(j + 1) * 5] = feat
            all_windows.append(X)

    X_train = np.array(all_windows, dtype=np.float32)
    print(f"[OK] LSTM-AE 训练数据: {X_train.shape}")

    model = build_lstm_autoencoder(input_dim=len(sensor_cols) * 5)
    print("[OK] 开始训练 LSTM-AE...")
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=1,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"[OK] LSTM-AE 模型已保存: {save_path}")

    detector = AnomalyLSTM(save_path)
    return detector, history


if __name__ == "__main__":
    # 快速验证
    if keras is None:
        print("TensorFlow 未安装，跳过 LSTM-AE 验证")
    else:
        model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "lstm_anomaly_model.h5")
        detector = AnomalyLSTM(model_path)

        if not detector.is_available:
            print("LSTM-AE 模型未训练，跳过验证（运行 main.py --train-lstm 来训练）")
        else:
            from data.sensor_simulator import generate_sensor_data
            df = generate_sensor_data()
            for dev_id in df["device_id"].unique():
                dev_df = df[df["device_id"] == dev_id].sort_values("timestamp")
                result = detector.detect(dev_df)
                print(f"设备 {dev_id}: anomaly_score={result['anomaly_score']:.6f}, is_anomaly={result['is_anomaly']}")
            print("LSTM-AE 验证通过")
