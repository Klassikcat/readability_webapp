import os
# uncomment the line below for postgres database url from environment variable
# postgres_local_base = os.environ['DATABASE_URL']
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'SECRET')
    DEBUG = False

domain_list = {
    '연설': 1,
    '편지': 2,
    '소설': 3,
    '에세이': 4,
    '자기소개서': 5
}

language_list = {
    'English': 1,
    '한국어': 2
}
