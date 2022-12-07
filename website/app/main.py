from flask import Flask, request, jsonify, redirect, url_for, render_template, session
from torch_utils import get_encoding, get_prediction
from flask_session import Session
import torch

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.form['comment']:
            session['comment'] = request.form['comment']
            # print(session.get('comment'))
            return redirect(url_for('predict'))
    else:
        return render_template('home.html')


@app.route('/predict')
def predict():
    # get sentence, tokenize, predict, return prob
    # print('=====================')
    # print(session.get('comment'))
    # print('======================')
    sentence = session.get('comment')
    input_ids, attention_mask = get_encoding(sentence)
    prob = get_prediction(input_ids, attention_mask)
    mbert_response = prob.tolist()
    print(mbert_response)

    label = torch.argmax(torch.tensor(mbert_response)).item()
    classes = {
        0: 'Hate', 1: 'Normal', 2: 'Offensive'
    }
    mbert_result = f'mBERT prediction:- {classes[label]} with probability {mbert_response[0][label]:.3f}'

    # print(response_obj)
    # except:
    #     response = jsonify({
    #         'error': 'something is wrong :('
    #     })
    return render_template('output.html', comment=sentence, mbert_result=mbert_result, mbert_response=mbert_response)


if(__name__ == '__main__'):
    app.run(debug=True)
