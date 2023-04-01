import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# global variable that will be used to store the interpreter
type_interpreter = None
labels = {0: "Type A", 1: "Type B", 2: "Type C"}

#defining function to read model from disk 
def input_type_classifier():
    global type_interpreter
    type_interpreter = tf.lite.Interpreter(model_path='Streamlit/classifier.tflite')
    type_interpreter.allocate_tensors()

def predict(image):
    #predicting output
    input_details = type_interpreter.get_input_details()
    output_details = type_interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']

    image = image.convert("RGB")
    image = image.resize((256, 256))
    img = np.array(image, dtype='float32')
    #normalize to 0 and 1
    img = img/255
    # reshape dataset to have a single 
    img = img.reshape((1, 256, 256, 3))
    type_interpreter.set_tensor(input_details[0]['index'], img)
    type_interpreter.invoke()
    predictions = type_interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(predictions[0])
    result = {
        'class': labels[pred]
    }
    
    return result

if __name__ == '__main__':
    file_uploaded = st.file_uploader("Upload the Image File", type=['jpg', 'jpeg', 'png'])
    
    if file_uploaded is not None:
        if type_interpreter is None:
            input_type_classifier()
        image = Image.open(file_uploaded)
        result = predict(image)
        col1, col2 = st.columns(2)
        col1.header("Type Classification Result")
        col1.write("The image is classified as "+result['class'])
