from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

db: SQLAlchemy = SQLAlchemy()
migrate = Migrate

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY'] = os.getenv('DATABASE_URL_ARTICLE')

    db.init_app(app)
    migrate.init_app(app, db)

    from readability_app.routes import main_route, predict_route, readability_route
    app.register_blueprint(main_route.bp)
    app.register_blueprint(predict_route.bp, url_prefix='/api')
    app.register_blueprint(readability_route.pb, url_prefix='/api')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)