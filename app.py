from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = 'my_model'
model = tf.keras.models.load_model(MODEL_PATH)

word_index = tf.keras.datasets.imdb.get_word_index()

def encode_text(text):
  tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], 250)[0]

def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred) 
  return result[0]

@app.route('/predict/', methods=['POST'])
def detect():
    review = request.json['review']
    result1 = predict(review)
    if result1[0] < 0.5:
        result = 'negative'
    else:
        result = 'positive'
    response = jsonify(result)
    return response


if __name__ == "__main__":
     app.run()

# if __name__ == "__main__":
#   app.run(host='0.0.0.0', port=8080)