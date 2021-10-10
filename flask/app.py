from flask import Flask
from flask import Flask, request, jsonify, abort, redirect, url_for, render_template, send_file,redirect,Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("main.html")

@app.route('/clusters')
def clusters():
    return render_template("clusters_map.html")

@app.route('/heatmap')
def heatmap():
    return render_template("heatmap.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
