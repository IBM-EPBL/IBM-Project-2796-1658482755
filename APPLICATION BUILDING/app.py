#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import numpy as np
from skimage.transform import resize
from keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model(r'C:\Users\MOWLITHARAN\Desktop\IBM\conversation engine for deaf and dumb\asl_model_84_54.h5')
#Initialize the Flask appfrom skimage.transform import resize
def detect(frame):
    frame=resize(frame,(64,64))
    x=image.img_to_array(frame)
    x=np.expand_dims(x,axis=0)
    pred=np.argmax(model.predict(x),axis=1)
    pred=list(model.predict(x)[0])
    index1=['A','B','C','D','E','F','G','H','I']
    inw=pred.index(max(pred))
    return inw
    
app = Flask(__name__,template_folder="Templates")
camera = cv2.VideoCapture(0)
def gen_frames():  
    while True:
        success, frame = camera.read()
        text=detect(frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
  
        # Use putText() method for
        # inserting text on video
        cv2.putText(frame, 
                    text, 
                    (50, 50), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
          # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)

