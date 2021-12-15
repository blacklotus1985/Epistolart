from flask import Flask, jsonify, request
import start

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!!!!</p>"

@app.route("/web",methods=['GET'])
def start_algo():
    return start.main()


if __name__=="__main__":
    app.run(debug=True)