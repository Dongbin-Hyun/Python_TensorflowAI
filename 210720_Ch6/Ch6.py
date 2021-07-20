import tensorflow as tf

#상수형
print("상수형")
x = tf.constant(3)
print(x)

y = tf.constant(3)
sess = tf.compat.v1.Session()
result = sess.run(y)
print(result)

#변수형
#변수 선언 : tf.value(초기값, name='이름')
print("변수형")

var_1 = tf.Variable(3)
var_2 = tf.Variable(10)

result_sum = var_1 + var_2
#Tensorflow에서 변수형은 그래프를 실행하기 전에 초기화 작업을 진행해야 함
#초기화 함수 추가
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
#초기화된 결과를 세션에 전달
sess.run(init)
print(sess.run(result_sum)) #에러코드 발생

#플레이스홀더
#선언 : tf.placeholder(dtype, shape=None, name=None)
#Ex) p_holder1 = tf.placeholder(dtype=tf.float32)
#dtype => 데이터타입 의미, 반드시 선언
print("플레이스홀더")

#학습용 데이터
print("학습용 데이터")

var_x = 15
var_y = 8

p_holder1 = tf.compat.v1.placeholder(dtype=tf.float32)
p_holder2 = tf.compat.v1.placeholder(dtype=tf.float32)

#학습용 데이터를 계산할 그래프
p_holder_sum = p_holder1 + p_holder2

sess = tf.compat.v1.Session()

#학습용 데이터를 넣기위한 feed_dict
result = sess.run(p_holder_sum, feed_dict = {p_holder1:var_x, p_holder2:var_y})
print(result)

A = [1, 3, 5, 7, 9]
B = [2, 4, 6, 8, 10]

ph_A = tf.compat.v1.placeholder(dtype=tf.float32)
ph_B = tf.compat.v1.placeholder(dtype=tf.float32)

result_sum = ph_A + ph_B

sess = tf.compat.v1.Session()
result = sess.run(result_sum, feed_dict={ph_A:A, ph_B:B})
print(result)



