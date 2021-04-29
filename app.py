from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
loaded_model = tf.keras.models.load_model('biLSTM_w2v.h5')
def clean_text(text):
  text = text.lower()
  new_text = re.sub('[^a-zA-z0-9\s]','',text)
  new_text = re.sub('rt', '', new_text)
  return new_text

num_classes = 5

embed_num_dims = 300

max_seq_len = 500

class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})
# routes
@app.route('/', methods=['POST'])
#@crossdomain(origin='*')
def predict():
    
    # get data
    data = request.get_json(force=True)
    sent = data['comment'][i]
    res={}
    res['after']=sent
    #message = clean_text(sent)
    #tokenizer = Tokenizer()
    #tokenizer.fit_on_texts(message)
    #seq = tokenizer.texts_to_sequences(message)
    #padded = pad_sequences(seq, maxlen=max_seq_len)
    #predictions = loaded_model.predict(padded)
    #output=class_names[np.argmax(predictions)]
    #res['prediction']=output
    #return jsonify(res)
    return jsonify(res)
if __name__ == "__main__":
    app.run(port = 5000, debug=True)
