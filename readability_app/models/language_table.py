from readability_app import db

class Language(db.Model):
    __tablename__ = 'language'

    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.VARCHAR(256), nullable=False)
    type = db.Column(db.String(), nullable=False)


    def __repr__(self):
        return f"Language('{self.id}', '{self.name}', '{self.type}')"