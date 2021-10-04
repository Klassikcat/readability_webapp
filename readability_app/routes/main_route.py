from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd
from readability_app.utils.readability_pred import predict_text
from readability_app.models import text_table, domain_table, language_table
from readability_app import db

bp = Blueprint('main', __name__)

#Landing Page
@bp.route('/')
def index():
    return render_template('index.html')

# English Readability Page
@bp.route('/english', methods=['GET','POST'])
def eng_index():
    if request.method == 'GET':
        text_orig = None
        pred_y, words_len, rot, conj, voca_div, longest_word, unique_word = "표시할 것이 없습니다.", "표시할 것이 없습니다.", "표시할 것이 없습니다.", "표시할 것이 없습니다.", "표시할 것이 없습니다.", "표시할 것이 없습니다.", "표시할 것이 없습니다.",
    elif request.method == 'POST':
        text_orig = request.form['text_name']
        domain_orig = request.form['domain_name']
        language = 1

        domain_list = {
            '연설': 1,
            '편지': 2,
            '소설': 3,
            '에세이': 4,
            '자기소개서': 5
        }

        domain_number = domain_list[domain_orig]

        pred_y, words_len, rot, conj, voca_div, longest_word, unique_word = predict_text(text_orig)
        pred_y, words_len, rot, conj, voca_div, longest_word, unique_word = float(pred_y), float(words_len), float(rot), int(conj), float(voca_div), int(longest_word), int(unique_word)
        table_txt_add = text_table.Text(txt=text_orig, readability=pred_y ,length=words_len,
                                        rot=rot, conjunction=conj, word_div=voca_div,
                                        longest=longest_word, rare_word=unique_word,
                                        lang_id=language, domain_id=domain_number)

        db.session.add(table_txt_add)
        db.session.commit()
    return render_template('editor.html', text=text_orig,
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

