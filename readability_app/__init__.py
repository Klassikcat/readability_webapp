from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///orm_db.sqlite3'

    from readability_app.models import domain_table, text_table, language_table
    db.init_app(app)
    migrate.init_app(app, db)

    from readability_app.routes import (main_route, english_route, reference_route)
    app.register_blueprint(main_route.bp)
    app.register_blueprint(english_route.bp)
    app.register_blueprint(reference_route.bp, url_prefix='/api')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)