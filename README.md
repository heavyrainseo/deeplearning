# deeplearning
인공지능과 머신러닝

## 선형회귀  
기초 수식
- 직선의 방정식 : $y = ax + b$
- MSE(또는 cost) : $cost = \frac{1}{n}\Sigma (y_i-\hat y)^2$
- 편미분
  - $\frac{\partial cost}{\partial a} = -\frac {2}{n} \Sigma x_i(y_i-(ax_i+b))$
  - $\frac{\partial cost}{\partial b} = -\frac {2}{n} \Sigma (y_i-(ax_i+b))$
- 학습 : $a = a-\alpha \frac{\partial cost}{\partial a}, b = b-\alpha \frac{\partial cost}{\partial a}, \alpha(learning \ rate) < 1$