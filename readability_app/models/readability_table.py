from readability_app import db

class Readability(db.Model):
    __tablename__ = 'readability'

    id = db.Column(db.Integer(), primary_key=True)
    text_id = db.Column(db.Integer(), db.ForeignKey('text.id'))
    text_info_id = db.Column(db.Integer(), db.ForeignKey('text_info.id'))
    text = db.Column(db.String(256), db.ForeignKey('text.text'))
    readability = db.Column(db.Float(32), nullable=False)