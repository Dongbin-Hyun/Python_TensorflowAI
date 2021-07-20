#학습에 필요한 모듈 선언
import tensorflow as tf
import numpy as np
#파이썬으로 과학 계산을 하려면 꼭 필요한 패키지.
# 다차원 배열을 위한 기능과 선형 대수 연산과 푸리에 변환 같은 고수준 수학 함수와 유사 난수 생성기를 포함
from matplotlib import pyplot as plt
# 대표적인 과학 계산용 그래프 라이브러리.
# 선 그래프, 히스토그램, 산점도 등을 지원하며 고품질 그래프를 그려준다.



#환경설정

#훈련용 데이터 수 선언
trainDataNumber = 100
#모델 최적화를 위한 학습률 선언
learingRate = 0.01
#총 학습 횟수 선언
totalStep = 1001

#[빌드단계]

#Step 1) 학습 데이터 준비

#항상 같은 난수를 생성하기 위해 시드설정
np.random.seed(321)

#학습 데이터 리스트 선언
xTrainData = list()
yTrainData = list()

#학습 데이터 생성
xTrainData = np.random.normal(0.0, 1.0, size=trainDataNumber)

for x in xTrainData:
    # y 데이터 생성
    y = 10 * x + 3 + np.random.normal(0.0, 3)
    yTrainData.append(y)

#학습 데이터 확인
plt.plot(xTrainData, yTrainData, 'bo')
plt.title("Train Data")
plt.show()

#[빌드단계]
#Step 2) 모델 생성을 위한 변수 초기화

#Weight 변수 선언
W = tf.Variable(tf.random.uniform([1]))
#Bias 변수 선언
b = tf.Variable(tf.random.uniform([1]))

#학습데이터 xTrainData가 들어갈 플레이스홀더 선언
#X : 학습데이터 x(독립변수)가 들어감
X = tf.compat.v1.placeholder(tf.float32)
#학습데이터 yTrainData가 들어갈 플레이스홀더 선언
#Y : 학습데이터 x의 실제값(정답데이터)가 들어감
Y = tf.compat.v1.placeholder(tf.float32)

#[빌드단계]
#Step 3) 학습 모델 그래프 구성

#3-1 학습데이터를 대표하는 가설 그래프 선언
#방법1 : 일반 연산기호를 이용하여 가설 수식 작성
hypothesis = W * X + b
#방법2 : tensorflow 함수를 이용하여 가설 수식 작성
#hypothesis = tf.add(tf.multiply(W,X),b)

#3-2 비용함수(오차함수, 손실함수) 선언
costFunction = tf.reduce_mean(tf.square(hypothesis - Y))

#3-3 비용함수의 값이 최소가 되도록 하는 최적화함수 선언
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learingRate)
train = optimizer.minimize(costFunction)

#[실행단계]
#학습 모델 그래프를 실행

#실행을 위한 세션 선언
sess = tf.compat.v1.Session()
#최적화 과정을 통하여 구해질 변수 W, b 초기화
sess.run(tf.compat.v1.global_variables_initializer())

#비용함수 그래프를 그리기 위한 변수 선언
WeightValueList = list()
costFunctionValueList = list()

print("-------------------------------------------------------------------------------------")
print("Train(Optimization) Start")
#totalStep 횟수 만큼 학습
for step in range(totalStep):
    #X, Y에 학습 데이터를 입력하여 비용함수, W, b, train 실행
    cost_val, W_val, b_val, _ = sess.run([costFunction, W, b, train], feed_dict={X: xTrainData, Y: yTrainData})

    #학습 결과값 저장
    WeightValueList.append(W_val)
    costFunctionValueList.append(cost_val)

    #학습 50회마다 중간 결과 출력
    if step % 50 == 0:
        print("Step : {}, cost : {}, W : {}, b : {}".format(step, cost_val, W_val, b_val))

        #학습 100회마다 중간 결과 Fitting Line 추가
        if step % 100 == 0:
            plt.plot(xTrainData, W_val * xTrainData + b_val, label='Step : {}'.format(step), linewidth=0.5)

print("Train Finished")

print("-------------------------------------------------------------------------------------")
print("[Train Result]")
#최적화가 끝난 학습 모델의 비용함수 값
cost_train = sess.run(costFunction, feed_dict={X: xTrainData, Y: yTrainData})

#최적화가 끝난 W, b 변수의 값
w_train = sess.run(W)
b_train = sess.run(b)
print("Train cost : {}, W : {}, b : {}".format(cost_train, w_train, b_train))
print("-------------------------------------------------------------------------------------")
print("[Test Result]")
#테스트를 위하여 x값 선언
testXValue = [2.5]
#최적화된 모델에 x에 대한 y값 계산
resultYValue = sess.run(hypothesis, feed_dict={X: testXValue})
#테스트 결과값 출력
print("x value is {}, y value is {}".format(testXValue, resultYValue))

#matplotlib를 이용하여 결과를 시각화

#결과 확인 그래프
plt.plot(xTrainData, sess.run(W) * xTrainData + sess.run(b), 'r', label='Fitting Line', linewidth=2)
plt.plot(xTrainData, yTrainData, 'bo', label='Train data')
plt.legend()
plt.title("Train Result")
plt.show()

#비용함수 최적화 그래프
plt.plot(WeightValueList, costFunctionValueList)
plt.title("costFunction curve")
plt.xlabel("Weight")
plt.ylabel("costFunction value")
plt.show()

#세션 종료
sess.close()








