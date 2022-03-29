import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Fashion MNIST 데이터셋 임포트
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 2. 데이터 전처리
train_images=train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images, test_images = train_images / 255.0, test_images / 255.0


# 3. 모델 구성
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
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