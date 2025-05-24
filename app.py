from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline,CustomData
import os
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            data = CustomData(
                age=int(request.form.get("age")),
                bmi=float(request.form.get("bmi")),
                children=int(request.form.get("children")),
                sex=request.form.get("sex"),
                smoker=request.form.get("smoker"),
                region=request.form.get("region"),
            )

            pred_df = data.get_data_as_datagrame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template("home.html", results=results[0])
        except Exception as e:
            print("Error during prediction:", e)
            return render_template("home.html", error=str(e))
