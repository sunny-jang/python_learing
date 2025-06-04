import numpy as np

from utils import sigmod, numerical_derivative


class LogicGate:
    def __init__(self, gate_name, xdata, tdata):
        self.name = gate_name

        # 입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        # 가중치 w, 바이어스 b 초기화
        self.__w = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        # 학습률
        self.__learning_rate = 1e-2

    # 손실함수
    def __loss_func(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmod(z)

        # cross-entropy
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log((1 - y) * delta))

    # 수치미분을 이용하여 손실함수가 최소가 될 때까지 학습시키는 함수

    def train(self):

        f = lambda x: self.__loss_func()
        print("initial error value = ", self.__loss_func())

        for step in range(100000):
            self.__w -= self.__learning_rate * numerical_derivative(f, self.__w)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)

            if step % 400 == 0:
                print("step", step, "error value = ", self.__loss_func())

    def predict(self, input_data):
        z =  np.dot(input_data, self.__w) + self.__b
        y = sigmod(z)

        if y > 0.5:
            result = 1
        else:
            result = 0

        return y, result


xdata = np.array([[0,0], [0,1], [1,0], [1,1]])
tdata = np.array([0,0,0,1])

AND_obj = LogicGate("AND", xdata, tdata)
AND_obj.train()

test_data = ([[0,0], [0,1], [1,0], [1,1]])

for inputdata in test_data:
    (sigmod_val, logical_val) = AND_obj.predict(inputdata)
    print(inputdata, " = ", logical_val)