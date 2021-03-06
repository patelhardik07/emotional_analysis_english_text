from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
loaded_model = tf.keras.models.load_model('Lstm_english .h5')
def clean_text(text):
  text = text.lower()
  new_text = re.sub('[^a-zA-z0-9\s]','',text)
  new_text = re.sub('rt', '', new_text)
  return new_text

num_classes = 5

embed_num_dims = 300

max_seq_len = 500

class_names = ['Joy', 'Fear', 'Anger', 'Sadness', 'Neutral']

data_train = pd.read_csv('data_train.csv', encoding='utf-8')
data_test = pd.read_csv('data_test.csv', encoding='utf-8')
dat = data_train.append(data_test, ignore_index=True)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dat['Text'])

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
    res={}
    for i in (data['comment']):
      sent = data['comment'][i]
      cmt=sent
      message = clean_text(sent)
      seq = tokenizer.texts_to_sequences([message])
      padded = pad_sequences(seq, maxlen=max_seq_len)
      predictions = loaded_model.predict(padded)
      pr=np.argmax(predictions)
      output=class_names[pr]
      res[i]={}
      res[i]['Comment']=cmt
      res[i]['Emotion']=output
    return jsonify(res)
if __name__ == "__main__":
    app.run(port = 5000, debug=True)
