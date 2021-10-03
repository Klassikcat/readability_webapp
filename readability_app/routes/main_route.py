from flask import Blueprint, render_template, request
import pandas as pd
from readability_pred import CLRDataset, preprocessing, cal_read_o_time, cal_total_read_o_time, predict_text

bp = Blueprint('main', __name__)

#Landing Page
@bp.route('/')
def index():
    return render_template('index.html')

# English Readability Page
@bp.route('/english', methods=['POST'])
def eng_index():
    text = request.GET.get('text')
    text = pd.DataFrame(data=[text], columns=['excerpt'])
    data = preprocessing(text)
    data = CLRDataset(data).get_df()
    data['paragraph_avg_rot'] = cal_total_read_o_time(data, '\n')
    data['sentence_avg_rot'] = cal_total_read_o_time(data, '. ')
    data['total_avg_rot'] = [cal_read_o_time(i) for i in data['excerpt']]
    pred_y = predict_text(text)
    context = {
        'text_data': data,
        'readability': pred_y
    }

    return render_template(request, 'editor.html', context)

@bp.route('/korean')
def kor_index():
    return render_template('under_construction.html')

@bp.route('/references', methods=['GET', 'DELETE'])
def ref_index():
    return render_template('refernces.html')

if __name__ == '__main__':
    app.run()
