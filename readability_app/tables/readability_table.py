from readability_app import db

class Readability(db.Model):
    __tablename__ = 'readability'

    id = db.Column(db.Integer(), primary_key=True)
    type = db.Column(db.String(256), nullable=False)
    readability = db.Column(db.Float(32), nullable=False)
    grade_level = db.Column(db.Float(32), nullable=False)
    text = db.relationship('text', back_popluation='text', cascale='all, delete')
    text_len = db.relationship('text_len', back_popluation='text_len', cascale='all, delete')
    text_conj = db.relationship('text_conj', back_popluation='text_conj', cascale='all, delete')
    text_unique = db.relationship('text_unique', back_poulation='text_unique', cascade='all, delete')
    text_rare = db.relationship('text_rare', back_popluation='text_rare', cascade='all, delete')
    text_per_paragraph = db.relationship('text_per_paragraph', back_poplulation='text_per_paragraph', cascade='all, delete')