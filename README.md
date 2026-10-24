# DeepLearning
인공지능과 머신러닝

## 선형회귀  
기초 수식
 - 직선의 방정식 : $y = ax + b$
 - MSE(또는 cost) : $cost = \frac{1}{n}\sum_{i=1}^{n} (y_i-\hat y_i)^2$
 - 편미분
   - $\frac{\partial cost}{\partial a} = -\frac {2}{n} \sum_{i=1}^{n} x_i\big(y_i-(a x_i+b)\big)$
   - $\frac{\partial cost}{\partial b} = -\frac {2}{n} \sum_{i=1}^{n} \big(y_i-(a x_i+b)\big)$
 - 학습 : $a = a-\alpha \frac{\partial cost}{\partial a},\; b = b-\alpha \frac{\partial cost}{\partial b},\; 0 < \alpha\;(\text{learning rate}) < 1$

## 로지스틱 회귀
- sigmoid : $\text{sigmoid}(z) = \frac{1}{1+e^{-(a x + b)}}$
 - cost : $cost = -\frac{1}{n}\sum_{i=1}^{n}\big( y_i \log \hat y_i + (1- y_i) \log (1-\hat y_i) \big)$

## 기타 수식
- 상관계수 r : $r=\frac{\operatorname{Cov}(x,y)}{\sigma_{x}\sigma_{y}}$
- 상관비(eta) : $\eta = \sqrt{\frac{SS_{between}}{SS_{total}}} = \sqrt{\frac{\sum_{k} n_k(\bar y_k-\bar y)^2}{\sum_{i} (y_i-\bar y)^2}},\; 0 \le \eta \le 1$
