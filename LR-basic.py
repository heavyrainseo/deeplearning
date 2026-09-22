import numpy as np


# 입력 데이터와 목표값
x = np.array([2, 4, 6, 8], dtype=float)
y = np.array([81, 93, 91, 97], dtype=float)

# 초기값과 학습률
a = 0.0
b = 0.0
learning_rate = 0.01
epochs = 10000

n = len(x)

for epoch in range(epochs):
    prediction = a * x + b
    error = y - prediction

    # README의 비용함수와 편미분 식
    cost = np.mean(error**2)
    gradient_a = -(2 / n) * np.sum(x * error)
    gradient_b = -(2 / n) * np.sum(error)

    # 경사하강법으로 a와 b 갱신
    a -= learning_rate * gradient_a
    b -= learning_rate * gradient_b

    if (epoch + 1) % 1000 == 0:
        print(f"epoch: {epoch + 1:5d}, cost: {cost:.6f}, a: {a:.6f}, b: {b:.6f}")

print("\n최종 결과")
print(f"직선의 방정식: y = {a:.6f}x + {b:.6f}")
print(f"최종 비용(MSE): {np.mean((y - (a * x + b))**2):.6f}")
print("예측값:", a * x + b)