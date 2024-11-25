from flask import FLask,request,rener_template

app=FLask(__name__)

@app.route("/")
def index():
    return rener_template("index.html")