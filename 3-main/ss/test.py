import tensorflow as tf
import numpy as np
import os

# 設定當前腳本的根目錄路徑
root_path = os.path.abspath(os.path.dirname(__file__))

# 拼接 validation.npz 文件路徑
validation_file_path = os.path.join(root_path, "dataset", "validation.npz")

# 檢查文件是否存在
if not os.path.exists(validation_file_path):
    raise FileNotFoundError(f"Validation file not found: {validation_file_path}")

# 加載 validation 資料
validation_data = np.load(validation_file_path)
validation_features = validation_data['data']
validation_labels = validation_data['label']

# 將標籤進行 one-hot 編碼
num_classes = 5  # 根據您的分類數進行調整
validation_labels_one_hot = tf.keras.utils.to_categorical(validation_labels, num_classes=num_classes)

# 拼接模型文件路徑
model_path = os.path.join(root_path, 'YOURMODEL.h5')

# 檢查模型文件是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# 加載保存的模型
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from '{model_path}'")
except Exception as e:
    raise RuntimeError(f"無法加載模型: {e}")

# 對模型進行 compile 操作
try:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled successfully.")
except Exception as e:
    raise RuntimeError(f"模型編譯失敗: {e}")

# 創建驗證資料集
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels_one_hot))
validation_dataset = validation_dataset.batch(64)

# 計算準確度
try:
    loss, accuracy = model.evaluate(validation_dataset)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    raise RuntimeError(f"評估模型時發生錯誤: {e}")
