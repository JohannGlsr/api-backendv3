from flask import Flask, request, jsonify
from keras.models import load_model
import pickle

# Chargement du tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Chargement du modèle
loaded_model = load_model('advance_model.h5')

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    phrase = request.form['phrase']
    sequence = phrase
    sequence = tokenizer.texts_to_sequences([sequence])
    while len(sequence[0]) < 35:
        sequence[0].insert(0, 0)
    prediction = loaded_model.predict(sequence)
    stringprediction = str(prediction[0][0])
    return jsonify({'prediction': stringprediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
