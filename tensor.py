import tensorflow as tf

텐서 = tf.constant([3,4,5]);
텐서2 = tf.constant([6,7,8]);


텐서3= tf.constant([[1,2],
                   [3,4]]);

print(tf.zeros([2,2]));
print(tf.matmul)

tall = 170
shoe = 260

a = tf.Variable(0.1)
b = tf.Variable(0.2)

#tf.keras.optimizers -> 경사하강법을 도와주는 함수
#Admam = w값을 일정하게 업데이트 하는게 아니고 경우에 따라 맞게 업데이트 해줌(gradient 자종 조절)
#러닝레이트 = 한번에 얼마큼 w 를 업데이트 할거냐.
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

def 손실함수():
    #return 실제값 - 예측값
    예측값 = tall * a + b
    return tf.square(260 - (예측값));

# 경사하강법을 실행시키는 함수
# 두가지의 함수를 넣어야 한다.
# opt.minimize(손실함수, var_list=[a,b])

for i in range(100):
    with tf.GradientTape() as tape:
        loss = 손실함수()

    gradients = tape.gradient(loss, [a, b])
    print(gradients)

    opt.apply_gradients(zip(gradients, [a, b]))

    print(f"손실 값: {loss.numpy()}, a: {a.numpy()}, b: {b.numpy()}")
