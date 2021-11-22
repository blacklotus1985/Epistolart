from flask import Flask

#pip install https://github.com/aboSamoor/pycld2/zipball/e3ac86ed4d4902e912691c1531d0c5645382a726 install this for pylcd2 wheel issues


app = Flask(__name__)


@app.route("/")
def hello_world():
  return "Hello, World!"