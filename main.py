#!/usr/bin/env python3.11

import warnings
warnings.filterwarnings("ignore")

from imageai.Classification import ImageClassification
import os
from pprint import pprint
import time

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

def init_classifiers():
  classifiers = {}

  for f in os.listdir("classification"):
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

def main():
  classifiers = init_classifiers()

  for f in os.listdir("img"):
    print("--------------------")
    print("Image: %s" % f)
    print("--------------------")
    for mname in classifiers:
      model = classifiers[mname]
      fsplit = os.path.splitext(f)
      if "jpg" in fsplit[1]:
        classify(model, os.path.join(os.getcwd(), "img", f))

if __name__ == '__main__':
  main()