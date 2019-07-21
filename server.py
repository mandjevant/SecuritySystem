
import numpy as np 
import cv2
import sys
import time
import os
from PIL import Image 
from flask import jsonify

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/detector/')
def detector():
	cascPath = path+r'\Classifiers\haarcascade_frontalface_alt_tree.xml'
	ymlLoc = path+r'\trainer\trainer.yml'
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read(ymlLoc)
	faceCascade = cv2.CascadeClassifier(cascPath);

	cam = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX

	while True:
		start_time = time.time()
		ret, frame =cam.read()
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
		for(x,y,w,h) in faces:
			nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
			cv2.rectangle(frame,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)

			if conf < 50:
				cv2.putText(frame,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 1.1, (0,255,0)) #Add your name
			else:
				cv2.putText(frame,str("Unknown " + '?')+"--"+str(conf), (x,y+h),font, 1.1, (0,255,0)) #Draw the text

		cv2.imshow('SecuritySystem',frame)
		
		#fps calculations
		print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()

	return render_template('index.html')

@app.route('/cameraList/')
def cameraList():
	# display number of cameras 
	index = 0
	arr = []
	while True:
	    cap = cv2.VideoCapture(index)
	    if not cap.read()[0]:
	        break
	    else:
	        arr.append(index)
	    cap.release()
	    index += 1
	
	# representation of webcams and numbers
	camDict = dict()
	for x in arr:
		if x == 0:
			camDict.update({x: "Webcam"})
		else:
			camDict.update({x: "USB cam " + str(x)})

	return jsonify(camDict)

@app.route('/faceRect/')
def faceRect():
	# frs on usb cam
	ff_d_cascPath = path+r'\Classifiers\haarcascade_frontalface_default.xml'
	pff_cascPath = path+r'\Classifiers\haarcascade_profileface.xml'
	ff_alt_tree_cascPath = path+r'\Classifiers\haarcascade_frontalface_alt_tree.xml'
	
	faceCascade = cv2.CascadeClassifier(ff_d_cascPath)
	profileCascade = cv2.CascadeClassifier(pff_cascPath)
	alt_treeCascade = cv2.CascadeClassifier(ff_alt_tree_cascPath)

	cam = cv2.VideoCapture(0)

	while(True):
		ret, frame = cam.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)		
		faces2 = profileCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
		faces3 = alt_treeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		for (x, y, w, h) in faces2:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		for (x, y, w, h) in faces3:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

		cv2.imshow('livefeed', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()

	return render_template('index.html')

@app.route('/ymlTrainer/')
def ymlTrainer():
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	cascPath = path+r'\Classifiers\haarcascade_frontalface_default.xml'
	dataPath = path+r'\dataSet'

	images, labels = img_and_lab(dataPath)
	cv2.imshow('test',images[0])
	cv2.waitKey(1)

	ymlLoc = path+r'\trainer\trainer.yml'

	recognizer.train(images, np.array(labels))
	recognizer.save(ymlLoc)
	cv2.destroyAllWindows()

	return render_template('index.html')

def img_and_lab(path):
	cascPath = path+r'\Classifiers\haarcascade_frontalface_default.xml'
	dataPath = dataSetGenerator(0)

	faceCascade = cv2.CascadeClassifier(cascPath)

	image_paths = [os.path.join(path, f) for f in os.listdir(path)]
	# images will contains face images
	images = []
	# labels will contains the label that is assigned to the image
	labels = []
	for image_path in image_paths:
    	# Read the image and convert to grayscal
		image_pil = Image.open(image_path).convert('L')
    	# Convert the image format into numpy array
		image = np.array(image_pil, 'uint8')
		# Get the label of the imagw
		nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
	 	#nbr=int(''.join(str(ord(c)) for c in nbr))
		# Detect the face in the image
		faces = faceCascade.detectMultiScale(image)
		# If face is detected, append the face to images and the label to labels
		for (x, y, w, h) in faces:
			images.append(image[y: y + h, x: x + w])
			labels.append(nbr)
			cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
			cv2.waitKey(10)
	# return the images list and labels list
	return images, labels

def dataSetGenerator(getal):
	cam = cv2.VideoCapture(getal)
	dataLoc = path+r'\dataSet'
	cascPath = path+r'\Classifiers\haarcascade_frontalface_default.xml'
	wrpath = path+r'\dataSet\face-'

	frontal_default_cascPath=cv2.CascadeClassifier(cascPath)
	#i=0
	offset=50
	name = input("Enter your id number ")

	for r in range(0,50):
		ret, frame = cam.read()
		gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces=frontal_default_cascPath.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
			cv2.imshow('img',frame[y-offset:y+h+offset,x-offset:x+w+offset])
			cv2.waitKey(10)

			cv2.imwrite(wrpath + name + '.' + str(r) + ".png", frame[y-offset:y+h+offset,x-offset:x+w+offset])
	cam.release()
	cv2.destroyAllWindows()

	return dataLoc


if __name__ == '__main__':
	path = os.path.dirname(os.path.abspath(__file__))
	app.run(debug=True)
