import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def load_model():
	model = tf.keras.models.load_model('./soil-classifier.keras')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Soil Type Classifier')

file = st.file_uploader("Upload an image of a soil", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['black_soil', 'cinder_soil', 'laterite_soil', 'peat_soil', 'yellow_soil']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)
