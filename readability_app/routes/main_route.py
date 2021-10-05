from flask import Blueprint, render_template, request, redirect, url_for
from readability_app.models import text_table
from readability_app.config import domain_list, language_list

bp = Blueprint('main', __name__)

#Landing Page
@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/korean')
def kor_index():
    return render_template('under_construction.html')

@bp.route('/reference')
def ref_index():
    text_ref = request.args.get('Text_search', type=str, default='')
    domain = request.args.get('Domain_search', type=str, default='')
    language = request.args.get('Language_search', type=str, default='')
    if text_ref:
        if request.method == 'GET':
            domain = domain_list[domain]
            language = language_list[language]

            measure_list = text_table.Text.query.filter(text_table.Text.domain_id == domain,
                                                        text_table.Text.lang_id == language,
                                                        text_table.Text.txt.ilike('%%{}%%'.format(text_ref))
                                                        ).all()

        return render_template('reference.html', measure_list=measure_list, text_ref=text_ref,
                               domain=domain_list.get(domain), language=language_list.get(language))
    else:
        return render_template('reference.html')

if __name__ == '__main__':
    app.run()

