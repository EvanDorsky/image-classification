#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

from imageai.Classification import ImageClassification
from imageai.Detection import ObjectDetection
import os
import time
from datetime import datetime
import json
from pprint import pprint
from subprocess import call, check_output
import subprocess

def classify(model, impath, result_count=10):
  print("====================")
  print(model["name"])
  print(model["desc"])
  start = time.time()

  predictions, probabilities = model["model"].classifyImage(impath, result_count)
  elapsed = time.time() - start
  print("elapsed time: %f msec" % (elapsed*1000))
  for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

def det_counts(detections):
  det_counts = {}
  for det in detections:
    if det["name"] not in det_counts:
      det_counts[det["name"]] = 0

    det_counts[det["name"]] += 1

  return det_counts

def detect(model, impath, result_count=10):
  print("====================")
  print(model["name"])
  print(model["desc"])
  start = time.time()

  # detections = model["model"].detectObjectsFromImage(input_image=impath, output_image_path=impath+"_"+model["name"]+".jpg", minimum_percentage_probability=20)
  detections = model["model"].detectObjectsFromImage(input_image=impath, minimum_percentage_probability=20)

  for det in detections:
    print(det["name"] , " : ", det["percentage_probability"], " : ", det["box_points"] )
    print("--------------------------------")

  dets = det_counts(detections)
  check_output(["exiftool", "-UserComment="+json.dumps(dets), impath, '-overwrite_original'])

  res = check_output(["exiftool", "-DateTimeOriginal", impath], text=True)
  # Parse the output to extract the datetime string
  datetime_str = res.split(': ')[1].strip()

  # Convert the string to a datetime object
  dt_obj = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')

  # Print the datetime as a timestamp
  print(int(dt_obj.timestamp()))

def init_classifiers():
  classifiers = {}

  for f in os.listdir("classification"):
    if f == ".DS_Store": continue
    fname_split = f.split('-')
    classifiers[fname_split[0]] = {
      "path": os.path.join(os.getcwd(), "classification", f),
      "name": fname_split[0]
    }

  for mname in classifiers:
    m = ImageClassification()
    if "mobilenet" in mname:
      m.setModelTypeAsMobileNetV2()
      classifiers[mname]["desc"] = "fastest prediction time and moderate accuracy"
    elif "resnet" in mname:
      m.setModelTypeAsResNet50()
      classifiers[mname]["desc"] = "fast prediction time and high accuracy"
    elif "inception" in mname:
      m.setModelTypeAsInceptionV3()
      classifiers[mname]["desc"] = "slow prediction time and higher accuracy"
    elif "densenet" in mname:
      m.setModelTypeAsDenseNet121()
      classifiers[mname]["desc"] = "slower prediction time and highest accuracy"

    m.setModelPath(classifiers[mname]["path"])
    m.loadModel()

    classifiers[mname]["model"] = m

  return classifiers

def init_detectors():
  detectors = {}

  for f in os.listdir("detection"):
    if f == ".DS_Store": continue
    detectors[f] = {
      "path": os.path.join(os.getcwd(), "detection", f),
      "name": f
    }

  for mname in detectors:
    d = ObjectDetection()
    if "retinanet" in mname:
      d.setModelTypeAsRetinaNet()
      detectors[mname]["desc"] = "high performance and accuracy, with longer detection time"
    elif "tiny-yolov3" in mname:
      d.setModelTypeAsTinyYOLOv3()
      detectors[mname]["desc"] = "moderate performance and accuracy, with a moderate detection time"
    elif "yolov3" in mname:
      d.setModelTypeAsYOLOv3()
      detectors[mname]["desc"] = "optimized for speed and moderate performance, with fast detection time"

    d.setModelPath(detectors[mname]["path"])
    d.loadModel()

    detectors[mname]["model"] = d

  return detectors

def main():
  classifiers = init_classifiers()
  detectors = init_detectors()

  for f in os.listdir("img"):
    print("--------------------")
    print("Image: %s" % f)
    print("--------------------")

    # detect
    for mname in detectors:
      model = detectors[mname]
      fsplit = os.path.splitext(f)
      if "jpg" in fsplit[1]:
        detect(model, os.path.join(os.getcwd(), "img", f))

    # classify
    # for mname in classifiers:
    #   model = classifiers[mname]
    #   fsplit = os.path.splitext(f)
    #   if "jpg" in fsplit[1]:
    #     classify(model, os.path.join(os.getcwd(), "img", f))

if __name__ == '__main__':
  main()