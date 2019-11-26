from flask import Blueprint,request,jsonify
import numpy as np
import  pandas as pd

app=Blueprint('post',__name__ )

@app.route("/jorawar" ,methods=['POST'])
def hello():
    theta=np.matrix([ 9.96328749,-3.58543356, 0.88523402])
    getdata = request.get_json()
    merged_mat_test = np.matrix([1,getdata["X1"] ,getdata["X2"]])
    print(theta.shape)
    print(merged_mat_test.shape)
    result = sigmod((np.dot(merged_mat_test, theta.T)))
    print("changes in flask")

    return str(np.round(result))

def sigmod(hip_cur):
    return 1 / (1 + np.exp(-hip_cur))