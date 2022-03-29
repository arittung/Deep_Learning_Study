import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Fashion MNIST 데이터셋 임포트
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 2. 데이터 전처리
# reshape : 첫번째 convolution은 모든 것을 포함한 single tensor를 예상하고 있기 때문 -> 60,000x28x28x1의 4D list로 만든다.
train_images=train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images, test_images = train_images / 255.0, test_images / 255.0


# 3. 모델 구성
# MaxPooling과 함께 convolution을 따라가다 보면 image는 highlight된 feature은 유지한 채 compress됨
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)), # Conv2D(생성하고 싶은 convolution의 수, Convolution 사이즈, Activation Function, nput data의 shape)
  tf.keras.layers.MaxPooling2D(2, 2), # MaxPooling을 2x2로 설정하면서 image의 사이즈는 1/4이 됨.
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
model.fit(train_images, train_labels, epochs=5)

# 6. 정확도 평가하기
loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
print("테스트 정확도 : ", accuracy)

# 7. 예측하기
randidx = np.random.randint(0,1000)
plt.imshow(test_images[randidx])
plt.show()

predictions = model.predict(test_images[randidx][np.newaxis,:,:])
#print(np.argmax(predictions))
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Andle boot']
print("예측 Fashion : ", class_names[np.argmax(predictions)])
