import tensorflow as tf
import pandas as pd
import numpy as np

# 엑셀같이 행과 열을 가진 파일을 열기위한 라이브러리
data = pd.read_csv('./gpascore.csv')

# print(data)


#데이터 전처리
#엑셀에 보면 간혹가다 빵꾸난 값들이 있음
#거기다가 평균값을 집어 넣던가, 행을 삭제하던가 한다.
#엑셀 하나하나 찾아서 하기 힘드니까, 판다스로 처리한다.
# print(data.isnull().sum())
data=data.dropna()
#빈 곳에 값 채우기
# data.fillna(100)
#gpa 다 출력하기
# print(data['gpa'])

#최솟값 구하기
# print(data['gpa'].min())
# print(data['gpa'].count())
# print(data['gpa'].max())
# print(data.isnull().sum())

#admit 데이터 값들만 잡아서 list로 만들어줌
y_data = data['admit'].values


x_data = []

# itterows를 사용하면 데이터프레임을 가로 한줄씩 출력해주라.
for i, rows in data.iterrows():
    #append로 추가
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])


# 딥러닝 만드는 방법
# 딥러닝 모델
model = tf.keras.models.Sequential([
    #Dense에 노드 갯수를 적어준다
    #기준이 없음 그냥 결과 잘 나올때까지 적늗나
    #근데 보통 2의 제곱수로 넣는다
    #근데 0,1로 예측할것이기 때문에 마지막 레이어는 Dense 1을 넣어준다.
    # 정수로는 예측이 어렵다. .
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    #시그모이드는 모든 값을 0과 1사이로 압축시켜주는 고마운 함수
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

#optimizer -> 경사하강법 같은 걸로 w값을 수정한다.
#기울기를 뺄때 learing_rate를 넣어서 빼긴 하는데, 기울기를 뺄때 빼는 값을 어떻게 정할지 선택하는 부분
#균등하게 빼면 러닝이 잘 안될 수도 있다.

#loss함수 -> 손실함수 결과값에서 얼마나 엇나갓는지를 측정
# 결과가 0과 1사이의 분류/확률 문제에서 쓰이는데
# binary_crossentropy 바이너리 크로스엔트로피 외우자!!
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#list를 넘파이 list로 만들어야 함
model.fit(np.array(x_data), np.array(y_data), epochs=1000)

#예측
예측값 = model.predict(np.array([[750, 3.70, 3],[400, 2.2, 1]]))
print(예측값)