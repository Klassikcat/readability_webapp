from readability_app import db

class Domain(db.Model):
    __tablename__ = 'domain'

    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(), nullable=False)