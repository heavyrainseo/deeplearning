# Seaborn Iris 데이터셋 다중 분류 모델
# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, utils
import matplotlib.pyplot as plt

# 2. 데이터 로드
df = sns.load_dataset('iris')
print("데이터 샘플:")
print(df.head())

# 3. 데이터 전처리
# 레이블(species) 인코딩 (문자열 -> 정수)
df['species'] = df['species'].astype('category').cat.codes

# 특성과 타겟 분리
X = df.drop('species', axis=1).values.astype(np.float32)
y = df['species'].values.astype(np.int32)

# 타겟 원-핫 인코딩
y_encoded = utils.to_categorical(y)

# 데이터 분할 (Train/Test)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(len(X) * 0.8)
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# 4. Keras 모델 구조 정의
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax') # 다중 분류를 위한 출력층
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 모델 학습 (EarlyStopping 포함)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)

# 6. 모델 평가 및 오차 행렬(Confusion Matrix) 시각화
# 분류 모델에서는 잔차 대신 혼동 행렬을 사용하여 예측 성능을 시각화합니다.
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# 혼동 행렬 직접 계산 (Numpy 활용)
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Setosa', 'Versicolor', 'Virginica'],
            yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
