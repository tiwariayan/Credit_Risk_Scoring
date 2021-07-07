# -*- coding: utf-8 -*-
"""
Created DateTime Tue, Jun 17 2020 12:10:00 

@author: Ayan Tiwari
"""

from flask import Flask, request, jsonify, render_template, Response
#from flask_cors import cross_origin
from creditscore import process_data



import pickle
import io

app = Flask(__name__)



@app.route("/creditscore", methods=["POST"])
def index():
    input_data_csv = request.files.get("inputDataFile")
    
    # print(request.headers["Content-Type"])
    
    if (input_data_csv == None):
        print("Form data: inputDataFile required")
        resp = create_error_json_message('Form data: inputDataFile required', 400)
        return resp

    else:
        with open("./temp/input.csv", "wb") as f:
            f.write(request.files.get('inputDataFile').read())
    
    try:
        process_data()
        with open("./temp/export.csv", "rb") as f:
            stream = f.read()
    
        res = Response(stream, mimetype="text/csv",
            headers={"Content-disposition":
                 "attachment; filename=export.csv"})
        return res
    except Exception as e:
        print(e)
        resp = create_error_json_message('Some error occurred while processing data!', 500)
        return resp
    
    return res

@app.route("/download", methods=["GET"])
def download():
    try:
        res = process_data()
    except Exception as e:
        print(e)
        resp = create_error_json_message('Some error occurred while processing data.', 500)
        return resp
    
    return res

def create_error_json_message(message, status_code):
    message = {
        'status': status_code,
        'message': message,
    }
    resp = jsonify(message)
    resp.status_code = status_code
    return resp
 

@app.route('/templates', methods=['POST', 'GET'])
def server_test():
    return "Server is running!"

@app.route('/help', methods=['GET','POST'])
#@cross_origin()
def help():
    return (
        "<br/><h1>Help: Customer Credit Score </h1> <br/><h2>HTTP Request:</h2> <br/><table style=width:50%><tr><td style='color:white;border:1px solid black;background-color:#325396;' valign=middle width=30% height=50px><span style=font-size: 20px>Post</span></td><td width=70% height=50px><span style=font-size: 20px>/businessCardParser</span></td></tr></table> <br/><h2>Description:</h2> <br/><table style=width:50% cellspacing=0><tr><td style='color:white;border:1px solid black;background-color:#325396;' valign=middle width=30% height=50px><span style=font-size: 20px>Parameter</span></td><td style='color:white;border:1px solid black;background-color:#325396;' valign=middle width=70% height=50px><span style=font-size: 20px>Value / Description</span></td></tr><tr><td style='border:1px solid black;' valign=middle width=30% height=60px><span style=font-size: 20px>businessCardImage</span></td><td style='border:1px solid black;' valign=middle width=70% height=60px><span style=font-size: 20px>Business card image as form file</span></td></tr><tr><td style='border:1px solid black;' valign=middle width=30% height=60px><span style=font-size: 20px>Subscription Key</span></td><td style='border:1px solid black;' valign=middle width=70% height=60px><span style=font-size: 20px>Azure Subscription key for Text Recognizer API</span></td></tr></table>")


#@cross_origin()
@app.route('/gui', methods=["GET"])
def gui():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
