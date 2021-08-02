from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageChops
import os




def ELA(img_path):
    """Performs Error Level Analysis over a directory of images"""
    
    TEMP = 'ela_' + 'temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)
        
    except:
        
        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)
        
       
    d = diff.load()
    
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    return diff


app = Flask(__name__)
uploaded_file = ''

app.config['image_upload'] = 'C:/Users/DeLL/Desktop/Fake video outputs/uploaded images'

@app.route('/main',methods = ['GET','POST'])
def main():
    return render_template('main.html')


@app.route('/upload_file' , methods = ['POST'])
def upload_file():

   #if request.method == 'POST':
   uploaded_file = request.files['data']

   uploaded_file.save(os.path.join(app.config['image_upload'],uploaded_file.filename))
   
   data=request.form['fav_language']
   data=data[0]
   print("I m on -> 58")
   model = load_model('new_model_casia.h5')
   print("succesfully imported model")
   '''if (data=='image_'):
       print("Image")
       model = load_model('new_model_casia.h5')
   elif (data=='deepfake_'):
       print("Deepfake")
       model = load_model('new_model_casia.h5')
   elif (data=='profile_'):
       print("Profile")
   else:
       print("else")
       return render_template('after.html','something')'''

   print("I m on -> 73")
   numpydata=np.array(ELA(uploaded_file).resize((128, 128))).flatten() / 255.0
   print("I m on -> 75")
   numpydata = np.resize(numpydata,(1,128,128,3))
   print("I m on -> 77")
   pred = model.predict(numpydata)
   
   
   print("We have saved pred")
   print("\n")
   print("Our prediction is = ",pred)
   
   
   return render_template('after.html',data = pred[0][0])




if __name__ == '__main__':
    app.run(host='0.0.0.0', port ='5000' , debug=True)