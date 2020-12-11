from flask import Flask, render_template, request
from flask import send_file,send_from_directory
import numpy as np
import cv2
import os
from os.path import dirname, join

app = Flask(__name__,static_folder='images')


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route("/changeresolution", methods=['GET'])
def getSuperResolution():
    filename = join(dirname(__file__), "models\EDSR_x4.pb")
    print(filename)
    return render_template("index.html")


@app.route("/download")
def download():
    return send_from_directory(directory='images', filename="upscaled.png")




@app.route("/changeresolution", methods=['POST'])
def postSuperResolution():
    image = request.files['data_file'].read()
    npimg = np.fromstring(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    model = request.form['model']
    sr_model = cv2.dnn_superres.DnnSuperResImpl_create()

    EDSR = join(dirname(__file__), "models\EDSR_x4.pb")
    ESPCN = join(dirname(__file__), "models\ESPCN_x4.pb")
    FSRCNN = join(dirname(__file__), "models\FSRCNN_x3.pb")
    LapSRN = join(dirname(__file__), "models\LapSRN_x8.pb")

    if(model == "EDSR"):
        sr_model.readModel(EDSR)
        sr_model.setModel("edsr", 4)
        upscaled = sr_model.upsample(img)

    if(model == "ESPCN"):
        sr_model.readModel(ESPCN)
        sr_model.setModel("espcn", 4)
        upscaled = sr_model.upsample(img)

    if(model == "FSRCNN"):
        sr_model.readModel(FSRCNN)
        sr_model.setModel("fsrcnn", 3)
        upscaled = sr_model.upsample(img)

    if(model == "LapSRN"):
        sr_model.readModel(LapSRN)
        sr_model.setModel("lapsrn", 8)
        upscaled = sr_model.upsample(img)

    (h,w,c) = img.shape
    original = "{h} x {w}".format(h=h,w=w)
    (uph,upw,upc) = upscaled.shape
    highres = "{h} x {w}".format(h=uph,w=upw)

    cv2.imwrite('SuperResolution/images/upscaled.png', upscaled)


    if not image:
        return "No file"
    #path = "sample.txt"
	# return send_file(upscaled)

    return render_template("index.html", original=original,
                           upscaled=highres)



if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
