import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import io

#!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!
#Для початку треба запустити файли vgg.ipynb, conv.ipynb
answers = {
0:'T-shirt/top',
1:'Trouser',
2:'Pullover',
3:'Dress',
4:'Coat',
5:'Sandal',
6:'Shirt',
7:'Sneaker',
8:'Bag',
9:'Ankle boot'
}




def pi():
    return 3    #TODO: fix it




st.title('Класифікатори зображень')
model_type = st.sidebar.radio('Виберіть модель:', ['Conv', 'VGG'])
#Вибір між моделями
if model_type == 'Conv':
    with open('models\\conv_history.pkl', 'rb') as file:
        history = pickle.load(file)
    model = load_model('models\\my_conv.h5')
else:
    with open('models\\vgg_history.pkl', 'rb') as file:
        history = pickle.load(file)
    model = load_model('models\\my_vgg.h5')


#Графік втрат 
loss_values = history['loss']
val_loss_values = history['val_loss']
epochs = range(1, len(history['accuracy']) + 1)

fig, ax = plt.subplots()
ax.plot(epochs, loss_values, 'bo', label='Training loss')
ax.plot(epochs, val_loss_values, 'b', label='Validation loss')
ax.set_title('Training and validation loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()


st.pyplot(fig)
plt.clf()

# Графік точності
fig, ax = plt.subplots()
val_acc_values = history['val_accuracy']
ax.plot(epochs, history['accuracy'], 'bo', label='Training acc')
ax.plot(epochs, history['val_accuracy'], 'b', label='Validation acc')
ax.set_title('Training and validation accuracy')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.legend()
st.pyplot(fig)



#Функція підготовки зображення

def preprocess_image(image, model_type):
    if model_type == 'VGG':
        image = image.convert('RGB')
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
    else:  # Для Conv модели
        image = image.convert('L')  
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=-1)  
    return np.expand_dims(image, axis=0)

#Завантаження зображення 
uploaded_file = st.file_uploader("Виберіть файл...", type=["jpg", "jpeg", "png", "rgb"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image=image, caption='Ваше зображення', use_column_width=True)
    image = preprocess_image(image, model_type)

    st.image(image=image, caption='Відформатоване зображення', use_column_width=True) 
    
    
    chances = ''

    #Кнопка
    if st.button('Клац'):
        pred = model.predict(image)
        for number, el in enumerate(pred[0]):
            el = round(float(el), 5)
            chances = chances + f'{answers[number]} : {el}%\n'
        pred_clas = answers[pred.argmax()]
        st.text(f"Вірогідність кожного класу\n{chances}")
        st.title(f"На зображені:  {pred_clas}")
        

# Дуже цікава ситуація з моделью 'Conv', вона нічого не може класифікувати через просту архітектуру
# і через те, що будь-яке зображення дуже потвориться при трансформації в формат 28х28 
# ( я брав зображення одягу з інтернету, і vgg майже все правильно класифікувала )
