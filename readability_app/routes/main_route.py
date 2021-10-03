from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd
from readability_app.utils.readability_pred import CLRDataset, preprocessing, cal_read_o_time, cal_total_read_o_time, predict_text

bp = Blueprint('main', __name__)

#Landing Page
@bp.route('/')
def index():
    return render_template('index.html')

# English Readability Page
@bp.route('/english', methods=['GET','POST'])
def eng_index():
    if request.method == 'GET':
        return render_template('editor.html')
    elif request.method == 'POST':
        text = request.form.get('text')
        return redirect(url_for('/result_eng'), text=text)


@bp.route('/result_eng', methods=['GET'])
def eng_res():
    if request.method == 'GET':
        #Preprocess
        text = request.form.get('text')
        text = pd.DataFrame(data=[text], columns=['excerpt'])
        data = preprocessing(text)
        data = CLRDataset(data).get_df()
        data['paragraph_avg_rot'] = cal_total_read_o_time(data, '\n')
        data['sentence_avg_rot'] = cal_total_read_o_time(data, '. ')
        data['total_avg_rot'] = [cal_read_o_time(i) for i in data['excerpt']]
        data = data.drop(columns=['excerpt', 'id', 'processed_exerpt'])

        #Data to Jinja Template
        pred_y = predict_text(text)
        words_len = data['num_words']
        rot = data['paragraph_avg_rot']
        conj = data['conjunction']
        voca_div = data['word_diversity']
        longest_word = data['longest_word']
        unique_word = data['unique_words']

    return render_template('result_eng.html',
                           readability=pred_y, words_len=words_len, read_o_time=rot,
                           conjunction=conj, voca_diverse=voca_div, longest_word=longest_word,
                           unique_words=unique_word)
@bp.route('/korean')
def kor_index():
    return render_template('under_construction.html')

@bp.route('/references', methods=['GET', 'DELETE'])
def ref_index():
    return render_template('refernces.html')

if __name__ == '__main__':
    app.run()

