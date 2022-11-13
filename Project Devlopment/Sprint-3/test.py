from tensorflow.keras.models import load_model
from keras.preprocessing import image

model=load_model("ECG.h5")
img=image.load_img("D:/Python/train I test/data/test/Normal/fig_2114.png",target_size=(64,64))

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=model.predict_classes(x)
pred

index=["Left Bundle Branch Block","Normal","Premature Atrial Contraction","Premature Ventricular Contractions","Right Bundle Branch Block","Ventricular Fibrillation"]
result=str(index[pred[0]])
result

