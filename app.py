import io

import numpy
from flask import Flask, render_template, url_for, redirect from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError from flask_bcrypt import Bcrypt
#from tensorflow import keras from flask import request
#from keras.models import load_model #from keras.preprocessing import image import numpy as np
import cv2 import os import sqlite3
from PIL import Image import pandas as pd import numpy as np
import matplotlib.pyplot as plt from tqdm import tqdm_notebook #%matplotlib inline
import tensorflow
print (tensorflow. version ) import tensorflow_addons as tfa import tensorflow.keras as keras from keras.models import load_model from PIL import Image
import numpy as np
generator_network = load_model('C:/Users/M Ashraf Nadeem/Desktop/FYP data/projcc/lr2hr_generator.h5')
adam_optimizer = tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5) generator_network.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[0.001, 1], optimizer=adam_optimizer)
# IMAGE_SIZE = [100, 100]
# model = load_model('model.h5') UPLOAD_FOLDER =os.path.join('static') app = Flask( name )
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db' app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app) bcrypt = Bcrypt(app) app.app_context().push()
login_manager = LoginManager() login_manager.init_app(app) login_manager.login_view = 'login'

global fileAddr
 


@login_manager.user_loader def load_user(user_id):
return User.query.get(int(user_id)) class User(db.Model, UserMixin):
id = db.Column(db.Integer, primary_key=True)
username = db.Column(db.String(20), nullable=False, unique=True) password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
username = StringField(validators=[
InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
password = PasswordField(validators=[
InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
submit = SubmitField('Register')

def validate_username(self, username): existing_user_username = User.query.filter_by(
username=username.data).first() if existing_user_username:
raise ValidationError(
'That username already exists. Please choose a different
one.')

class LoginForm(FlaskForm):
username = StringField(validators=[
InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
password = PasswordField(validators=[
InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
submit = SubmitField('Login')

# @app.route('/') # def home():
#	return render_template('home.html') @app.route('/', methods=['GET', 'POST']) def login():
form = LoginForm()
if form.validate_on_submit():
user = User.query.filter_by(username=form.username.data).first() if user:
if bcrypt.check_password_hash(user.password, form.password.data): login_user(user)
return redirect(url_for('index')) return render_template('login.html', form=form)
 
@app.route('/register', methods=['GET', 'POST']) def register():
form = RegisterForm()

if form.validate_on_submit():
hashed_password = bcrypt.generate_password_hash(form.password.data) new_user = User(username=form.username.data,
password=hashed_password)
db.session.add(new_user) db.session.commit()
return redirect(url_for('login'))

return render_template('register.html', form=form) @app.route('/index', methods=['GET', 'POST']) @login_required
def index():
return render_template('index.html')


@app.route('/logout', methods=['GET', 'POST']) @login_required
def logout():
logout_user()
return redirect(url_for('login'))

@app.route("/submit", methods=['GET', 'POST'])




def show_test_results():
if request.method == 'POST':
img = request.files['image'] img_name = "static/" + img.filename fileAdr = img_name
print(fileAdr) img.save(img_name)
#get_img = Image.open(img_name)
# files = np.random.choice(new_test, size=1) # low_quality_images = []
# high_quality_images = [] # for file in files:
img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) img2 = cv2.resize(img, (256, 256))
img3 = cv2.resize(img, (64, 64)) low_quality_images = (img3 - 127.5) / 127.5 print(low_quality_images.shape) high_quality_images = (img2 - 127.5) / 127.5 print(high_quality_images.shape) low_quality_images = np.array(low_quality_images) print(low_quality_images.shape)
high_quality_images = np.array(high_quality_images) print(high_quality_images.shape)
low_quality_images = low_quality_images.reshape((1, 64, 64, 3)) fake_high_quality_images = generator_network.predict(low_quality_images)
 

# print ("Real quality input images")
# plt.imshow((high_quality_images+1.0)/2.0) # plt.axis('off')
# plt.show()
low_quality_images = np.squeeze(low_quality_images, axis=0) #print("Low quality input images") #plt.imshow((low_quality_images + 1.0) / 2.0) #plt.axis('off')
#image1 = Image.fromarray(low_quality_images.astype('uint8')) print('low_quality_images', low_quality_images)
image1 = np.asarray((low_quality_images+1.0)/2.0) * 255 #image1=cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
#image1 = cv2.cvtColor(np.array(image1 * 255.0), cv2.COLOR_RGB2BGR) print('image1', image1.shape)
print('image1', image1)
#image1 = Image.fromarray(low_quality_images.astype('uint8')) #img_name1 = "static/low_quality.jpg" # Specify the desired file path
and name
#image1.save('static/image1.jpg') cv2.imwrite('static/image1.jpg', image1) #filadre2 = img_name1

# image1 = Image.fromarray(low_quality_images.astype('uint8')) # adress1=str(fileAdr) + str (1)
# img_name1 = os.path.join("static", adress1)

# Optionally, you can save the image to a file #plt.show()
fake_high_quality_images = np.squeeze(fake_high_quality_images, axis=0)

#print("Generated high quality images") #plt.imshow((fake_high_quality_images + 1.0) / 2.0)
#image2 = Image.fromarray(fake_high_quality_images.astype('uint8')) image2 =
cv2.cvtColor(numpy.array(((fake_high_quality_images+1.0)/2.0)*255), cv2.COLOR_RGB2BGR)
img_name2 = "static/fake_img.jpg" # Specify the desired file path and name
#image2.imsave('static/image2.jpg') cv2.imwrite('static/image2.jpg', image2) print('image2', image2)
#filadre = img_name2

# adress2=str(fileAdr)+str(2) #
# img_name2 = os.path.join("static", adress2)

# Optionally, you can save the image to a file plt.axis('off')

plt.show()
return render_template("pred.html", fake_images='static/image2.jpg', low_quality='static/image1.jpg')
 if  name == " main ":
 app.run(debug=True)
