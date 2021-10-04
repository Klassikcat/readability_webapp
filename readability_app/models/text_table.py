from readability_app import db

class Text(db.Model):
    __tablename__ = 'text'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer(), primary_key=True)
    txt = db.Column(db.String(128), nullable=False)
    readability = db.Column(db.Float(32), nullable=False)
    lang_id = db.Column(db.Integer(), db.ForeignKey('language.id'))
    domain_id = db.Column(db.Integer(), db.ForeignKey('domain.id'))
    length = db.Column(db.Integer())
    rot = db.Column(db.Float(), nullable=False)
    conjunction = db.Column(db.Integer())
    word_div = db.Column(db.Float())
    longest = db.Column(db.Integer())
    rare_word = db.Column(db.Integer())

    lang_rel = db.relationship('Language', backref='text_rel')
    domain_rel = db.relationship('Domain', backref='text_rel')