from readability_app import db

class Text(db.Model):
    __tablename__ = 'text'

    id = db.Column(db.Integer(), primary_key=True)
    text = db.Column(db.String(128), nullable=False, unique=True)
    text_len = db.Column(db.String(128), nullable=False, unique=False)
    text_conj = db.Column(db.Integer(), nullable=False)
    text_unique = db.Column(db.String(128))
    text_rare = db.Column(db.String(128))
    text_per_paragraph = db.Column(db.Float(32))
