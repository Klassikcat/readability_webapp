from flask import Blueprint, request

bp = Blueprint('english', __name__)

@bp.route('/english', methods=['POST'])
def add_text():
    request_data = request.form