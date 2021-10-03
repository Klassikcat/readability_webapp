from readability_app import db

class Text_info(db.Model):
    __tablename__ = 'text_info'

    id = db.Column(db.Integer(), primary_key=True)
    text_id = db.Column(db.Integer(), db.ForeignKey('text.id'))
    readability_id = db.Column(db.Integer(), db.ForeignKey('readability.id'))
    readability = db.Column(db.Float(), db.ForeignKey('readability.readability'))
    para_rot = db.Column(db.Float(), nullable=False)
    sent_rot = db.Column(db.Float(), nullable=False)
    tobe_verb = db.Column(db.Integer())
    auxverb = db.Column(db.Integer())
    conjunction = db.Column(db.Integer())
    pronoun = db.Column(db.Integer())
    nominalization = db.Column(db.Integer())
    pronoun_b = db.Column(db.Integer())
    interrogative = db.Column(db.Integer())
    article = db.Column(db.Integer())
    subordination = db.Column(db.Integer())
    conjunction_b = db.Column(db.Integer())
    preposition_b = db.Column(db.Integer())