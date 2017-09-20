# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
"""
Groups together code used for creating a NuPIC model and dealing with IO.
(This is a component of the One Hot Gym Anomaly Tutorial.)
"""
import importlib
import sys
import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.frameworks.opf.metrics import MetricSpec, MetricAAE
from nupic.frameworks.opf.prediction_metrics_manager import MetricsManager

import nupic_anomaly_output


DESCRIPTION = (
  "Starts a NuPIC model from the model params returned by the swarm\n"
  "and pushes each line of input from the gym into the model. Results\n"
  "are written to an output file (default) or plotted dynamically if\n"
  "the --plot option is specified.\n"
)

CC_NAME = "sim_fraud_small"
DATA_DIR = "."
MODEL_PARAMS_DIR = "./model_params"
# '2017-08-24 03:19:52'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# DATE_FORMAT = "%m/%d/%Y %H:%M"


def createModel(modelParams):
  """
  Given a model params dictionary, create a CLA Model. Automatically enables
  inference for amount.
  :param modelParams: Model params dict
  :return: OPF Model object
  """
  model = ModelFactory.create(modelParams)

  #In case of anomaly detection, this predicted field is somewhat optional
  model.enableInference({"predictedField": "isFraud"})
  return model

def getModelMetricsManager(model):

  metricSpecs = (
    MetricSpec(field='isFraud', metric='multiStep',
               inferenceElement='multiStepBestPredictions',
               params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
    MetricSpec(field='isFraud', metric='trivial',
               inferenceElement='prediction',
               params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
    MetricSpec(field='isFraud', metric='multiStep',
               inferenceElement='multiStepBestPredictions',
               params={'errorMetric': 'altMAPE', 'window': 1000, 'steps': 1}),
    MetricSpec(field='isFraud', metric='trivial',
               inferenceElement='prediction',
               params={'errorMetric': 'altMAPE', 'window': 1000, 'steps': 1}),
  )

  metricsManager = MetricsManager(metricSpecs,model.getFieldInfo(),
                                  model.getInferenceType())
  return metricsManager

def getModelParamsFromName(creditCardName):
  """
  Given a gym name, assumes a matching model params python module exists within
  the model_params directory and attempts to import it.
  :param creditCardName: Gym name, used to guess the model params module name.
  :return: OPF Model params dictionary
  """
  importName = "model_params.%s_model_params" % (
    creditCardName.replace(" ", "_").replace("-", "_")
  )
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. Run swarm first!"
                    % creditCardName)
  return importedModelParams



def runIoThroughNupic(inputData, model, creditCardName, plot):
  """
  Handles looping over the input data and passing each row into the given model
  object, as well as extracting the result object and passing it into an output
  handler.
  :param inputData: file path to input data CSV
  :param model: OPF Model object
  :param creditCardName: Gym name, used for output handler naming
  :param plot: Whether to use matplotlib or not. If false, uses file output.
  """
  inputFile = open(inputData, "rb")
  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  shifter = InferenceShifter()
  if plot:
    output = nupic_anomaly_output.NuPICPlotOutput(creditCardName)
  else:
    output = nupic_anomaly_output.NuPICFileOutput(creditCardName)

  metricsManager = getModelMetricsManager(model)

  counter = 0
  for row in csvReader:
    counter += 1
    if (counter % 100 == 0):
      print "Read %i lines..." % counter
    timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
    paytype = row[1]
    amount = float(row[2])
    nameOrig = row[3]
    oldbalanceOrig = float(row[4])
    newbalanceOrig = float(row[5])
    nameDest = row[6]
    oldbalanceDest = float(row[7])
    newbalanceDest = float(row[8])
    isFraud = int(row[9])
    isFraudCalc = int(row[10])

    # print("Row %i, amount read %f " %(counter, amount))
    result = model.run({
      "timestamp": timestamp,
      "paytype": paytype,
      "amount": amount,
      "nameOrig": nameOrig,
      "oldbalanceOrig": oldbalanceOrig,
      "newbalanceOrig": newbalanceOrig,
      "nameDest": nameDest,
      "oldbalanceDest": oldbalanceDest,
      "newbalanceDest": newbalanceDest,
      "isFraud": isFraud,
      "isFraudCalc": isFraudCalc
    })

    if plot:
      result = shifter.shift(result)

    metrics = metricsManager.update(result)
    # You can collect metrics here, or attach to your result object.
    result.metrics = metrics
    # print("After %i records, 1-step altMAPE=%f", counter,
    #       result.metrics["multiStepBestPredictions:multiStep:"
    #                      "errorMetric='altMAPE':steps=1:window=1000:"
    #                      "field=isFraud"]) 
    prediction = result.inferences["multiStepBestPredictions"][1]
    anomalyScore = result.inferences["anomalyScore"]
    output.write(timestamp, isFraud, prediction, anomalyScore)
    # output.write(timestamp, amount, prediction, anomalyScore)

  inputFile.close()
  output.close()



def runModel(creditCardName, plot=False):
  """
  Assumes the gynName corresponds to both a like-named model_params file in the
  model_params directory, and that the data exists in a like-named CSV file in
  the current directory.
  :param creditCardName: Important for finding model params and input CSV file
  :param plot: Plot in matplotlib? Don't use this unless matplotlib is
  installed.
  """
  print "Creating model from %s..." % creditCardName
  model = createModel(getModelParamsFromName(creditCardName))
  inputData = "%s/%s.csv" % (DATA_DIR, creditCardName.replace(" ", "_"))
  runIoThroughNupic(inputData, model, creditCardName, plot)



if __name__ == "__main__":
  print DESCRIPTION
  plot = False
  args = sys.argv[1:]
  if "--plot" in args:
    plot = True
  runModel(CC_NAME, plot=plot)
