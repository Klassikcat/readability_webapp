from flask import Blueprint, render_template, request, redirect, url_for
from readability_app.models import text_table
from readability_app import db
bp = Blueprint('reference', __name__)

@bp.route('/reference/<id>', methods=['GET', 'DELETE'])
def del_ref(id=None):
    if id != None:
        table_txt_del = text_table.Text.query.filter(text_table.Text.id == id).first()

        db.session.delete(table_txt_del)
        db.session.commit()

    return redirect(url_for('main.ref_index'))