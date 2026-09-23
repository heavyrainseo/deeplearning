# deeplearning
인공지능과 머신러닝

## 선형회귀  
기초 수식
- 직선의 방정식 : $y = ax + b$
- MSE(또는 cost) : $cost = \frac{1}{n}\Sigma (y_i-\hat y)^2$
- 편미분
  - $\frac{\partial cost}{\partial a} = -\frac {2}{n} \Sigma x_i(y_i-(ax_i+b))$
  - $\frac{\partial cost}{\partial b} = -\frac {2}{n} \Sigma (y_i-(ax_i+b))$
- 학습 : $a = a-\alpha \frac{\partial cost}{\partial a}, b = b-\alpha \frac{\partial cost}{\partial b}, 0 \lt \alpha(learning \ rate) < 1$

## 로지스틱회귀
기초 수식  
- sigmoid : $sig = \frac {1}{1+e
^{-(ax+b)}}$
- cost : $cost=-\frac {1}{n} \Sigma (y_ilog\hat y+(1- y_i)log(1-\hat y))$

## 기타 수식
- 상관계수 r : $r=\frac {Cov(x,y)}{\sigma _x \sigma _y}$
- 상관비(eta) : $\eta = \sqrt {SS _{between}/SS_{total}} = \sqrt {\Sigma _k n_k(\bar y_k-\bar y)^2 / \Sigma _i (y_i-\bar y)^2}, \ 0 \le \eta \le 1$