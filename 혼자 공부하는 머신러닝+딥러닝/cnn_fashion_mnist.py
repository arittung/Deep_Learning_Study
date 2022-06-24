from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

print("[*] data preprocessing")
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print("[*] model generation")
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape = (28,28,1)))
model.add(keras.layers.MaxPooling2D(2)) # feature map size= (14,14,32)
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2)) # feature map size= (7,7,64)
model.add(keras.layers.Flatten()) # feature map size= (7*7*64) = 3136
model.add(keras.layers.Dense(100, activation='relu')) # feature map size= 3136 * 100 + 100 = 313700
model.add(keras.layers.Dropout(0.4)) # 은닉층의 과대적합 막아 성능 개선
model.add(keras.layers.Dense(10, activation='softmax')) # 여기서 확률 계산
# feature map size= 100 * 10 + 10 = 1010

#keras.utils.plot_model(model)
#keras.utils.plot_model(model, show_shapes=True, to_file='cnn-fashion_mnnist-architecture.png', dpi=300)

# model compile & training
print("[*] model training..")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# ModelCheckpoint 콜백 : 최상의 검증 점수를 만드는 모델 저장.
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
# EarlyStopping 콜백 : 조기 종료
# restore_best_weights= True : 가장 낮은 검증 손실을 낸 모델 파라미터로 되돌림.
# patience : 검증 점수가 향상되지 않더라도 참을 에포크 횟수로 지정
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# loss graph
print("[*] loss graph")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# model evaluation
print("[*] model evaluation")
model.evaluate(val_scaled, val_target)

# new data prediction
print("[*] new data prediction")
plt.imshow(val_scaled[0].reshape(28,28), cmap='gray_r')
plt.show()
preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(1,11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

print("[*] new data answer")
classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Andle boot']
print(classes[np.argmax(preds)])

# test set prediction
print("[*] testset prediction Performance")
test_scaled = test_input.reshape(-1, 28, 28, 1)/ 255.0
model.evaluate(test_scaled, test_target)
