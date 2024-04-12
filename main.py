#!/usr/bin/env python3.11

import warnings
warnings.filterwarnings("ignore")

from imageai.Classification import ImageClassification
import os
from pprint import pprint
import time

def predict(model, impath, result_count=10):
  print("====================")
  print(model["name"])
  print(model["desc"])
  start = time.time()
  predictions, probabilities = model["model"].classifyImage(impath, result_count)
  elapsed = time.time() - start
  print("elapsed time: %f msec" % (elapsed*1000))
  for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

def main():
  models = {}
  for f in os.listdir("models"):
    fname_split = f.split('-')
    models[fname_split[0]] = {
      "path": os.path.join(os.getcwd(), "models", f),
      "name": fname_split[0]
    }

  for mname in models:
    m = ImageClassification()
    if "mobilenet" in mname:
      m.setModelTypeAsMobileNetV2()
      models[mname]["desc"] = "fastest prediction time and moderate accuracy"
    elif "resnet" in mname:
      m.setModelTypeAsResNet50()
      models[mname]["desc"] = "fast prediction time and high accuracy"
    elif "inception" in mname:
      m.setModelTypeAsInceptionV3()
      models[mname]["desc"] = "slow prediction time and higher accuracy"
    elif "densenet" in mname:
      m.setModelTypeAsDenseNet121()
      models[mname]["desc"] = "slower prediction time and highest accuracy"

    m.setModelPath(models[mname]["path"])
    m.loadModel()

    models[mname]["model"] = m

  for f in os.listdir("img"):
    print("--------------------")
    print("Image: %s" % f)
    print("--------------------")
    for mname in models:
      model = models[mname]
      fsplit = os.path.splitext(f)
      if "jpg" in fsplit[1]:
        predict(model, os.path.join(os.getcwd(), "img", f))

if __name__ == '__main__':
  main()