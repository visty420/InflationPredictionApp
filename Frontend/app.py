from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

login_manager = LoginManager()
login_manager.login_view = 'auth.login' 
login_manager.init_app(app)

db = SQLAlchemy(app)

from Server.models import User

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from Server.auth import auth as auth_blueprint
app.register_blueprint(auth_blueprint)

from main import main as main_blueprint
app.register_blueprint(main_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
