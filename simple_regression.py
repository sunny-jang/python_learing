import numpy as np
from utils import numerical_derivative

# 학습데이터 준비
x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
y_data = np.array([2, 3, 4, 5, 6]).reshape(5, 1)

print(x_data)

# 임의의 직선 y = Wx + b 정의 (임의의 값으로 가중치 W, 바이어스 b 초기화)
w = np.random.rand(1, 1)
b = np.random.rand(1)

print("W = ", w)
print("W.shape = ", w.shape)
print("b = ", b)
print("b.shape = ", b.shape)


# 손실함수 E(W,b) 정의
def loss_func(x, t):
    y = np.dot(x, w) + b

    return (np.sum((t - y) ** 2)) / (len(x))


# 학습을 마친 후, 임의의 데이터에 대해 미래값 예측 함수
# 입력변수 x: numpy type
def predict(x):
    y = np.dot(x, w) + b

    return y


learning_rate = 1e-2  # 발산하는 경우, ie-3~ 1e-6 등으로 바꾸어서 실행

f = lambda x: loss_func(x_data, y_data)

print("initial error value =", loss_func(x_data, y_data), "initial w = ", w, "initial b = ", b)

for step in range(10000):
    w -= learning_rate * numerical_derivative(f, w)
    b -= learning_rate * numerical_derivative(f, b)

    if step % 400 == 0:
        print("initial error value =", loss_func(x_data, y_data), "initial w = ", w, "initial b = ", b)


print(predict(43))