import tensorflow as tf
import nncdata as nd

nnc = nd.NncData()
x_train, x_test, y_train, y_test = nnc.load_nnc_image('smiledata.csv',width=150,height=150)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, kernel_size=(7,7), activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(30, kernel_size=(3,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Activation('tanh'))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(150, activation='relu'))

model.add(tf.keras.layers.Dense(50, activation='relu'))

model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=64)
sc = model.evaluate(x_test, y_test)
print("acc=",sc[1])

#show model
model.summary()

#save model(.h5)
model.save('smile_Lenet.h5')

