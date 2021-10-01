import readability_pred
from flask import Blueprint, render_template, request

bp = Blueprint('main', __name__)

#Landing Page
@app.route('/')
def index():
    return render_template('index.html')

# English Readability Page
@bp.route('/english')
def eng_index():


@bp.route('/references')
def ref_index():


if __name__ == '__main__':
    app.run()
