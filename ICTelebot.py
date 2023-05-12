from telegram.ext import Updater, Filters, CommandHandler, MessageHandler
import tensorflow as tf
from io import BytesIO
import cv2
import numpy as np

#Load training and testing data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

#Labels For Images in Cifar10
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#Creating Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#Train Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=11, validation_data=(x_test, y_test))
model.save('image_classifier.model')

#Create Commands
def start(updater, context):
    updater.message.reply_text("""Welcome to Kido's Test bot!
/commands for bot commands""")

def command(updater, context):
    updater.message.reply_text("""/start - show start message
/help - how to use bot
/commands - show bot commands""")

def help_(updater, context):
    updater.message.reply_text("Send an image you want me to classify")

#Handle Messages
def message(updater, context):
    msg = updater.message.text
    print(msg)
    updater.message.reply_text(msg)

#Handle Images
def image(updater, context):
    photo = updater.message.photo[-1].get_file()
    p = BytesIO(photo.download_as_bytearray())
    photo_bytes = np.asarray(bytearray(p.read()), dtype=np.uint8)

    img = cv2.imdecode(photo_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    p = model.predict(np.array([img / 255]))
    updater.message.reply_text(f"In this image I can see a/an {labels[np.argmax(p)]}")


#Connect with Telegram Bot
updater = Updater("5837595386:AAEMpK2i45dJGKHhL1B4V98s3DCjePlUpNM")
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("help", help_))
dispatcher.add_handler(CommandHandler("commands", command))

dispatcher.add_handler(MessageHandler(Filters.text, message))
dispatcher.add_handler(MessageHandler(Filters.photo, image))

updater.start_polling()
updater.idle()
