import os
import numpy as np
import tensorflow as tf
from flask import Flask,request,render_template,redirect,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app=Flask(__name__)
model=load_model('ECG.h5')


@app.route('/')
@app.route('/home.html')
def about():
    return render_template("home.html")
@app.route('/info.html')
def info():
    return render_template('info.html')
@app.route("/upload.html")
def test():
    return render_template("upload.html")
@app.route("/predict.html/<result>/<filepath>")
def test1(result,filepath):
    return render_template("predict.html",result=result,filepath=filepath)
@app.route("/Left_Bundle_Branch_Block.html")
def Left_Bundle_Branch_Block():
    return render_template("Left_Bundle_Branch_Block.html")
@app.route("/Premature_Atrial_Contraction.html")
def Premature_Atrial_Contraction():
    return render_template("Premature_Atrial_Contraction.html")
@app.route("/Premature_Ventricular_Contractions.html")
def Premature_Ventricular_Contractions():
    return render_template("Premature_Ventricular_Contractions.html")
@app.route("/Right_Bundle_Branch_Block.html")
def Right_Bundle_Branch_Block():
    return render_template("Right_Bundle_Branch_Block.html")
@app.route("/Ventricular_Fibrillation.html")
def Ventricular_Fibrillation():
    return render_template("Ventricular_Fibrillation.html")
@app.route('/upload.html',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']
        if f.filename=='':
            flash('No file selected')
        else:
            print("Analysing...")
            basepath=os.path.dirname('__file__')
            
            filepath=os.path.join(basepath,"static\\uploads",f.filename)
            f.save(filepath)
            
            img=tf.keras.utils.load_img(filepath,target_size=(64,64))
            x=tf.keras.utils.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            print("Predicting...")
            pred=model.predict(x)
            classes_x=np.argmax(pred,axis=1)
            print(classes_x)
            
            index=["Left Bundle Branch Block","Normal","Premature Atrial Contraction","Premature Ventricular Contractions","Right Bundle Branch Block","Ventricular Fibrillation"]
            result=str(index[classes_x[0]])
            print("Prediction Done...")

            return render_template('predict.html',result=result,filepath='static/uploads/'+f.filename)
    return None
if __name__=="__main__":
    
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    
