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

import nupic_anomaly_output


DESCRIPTION = (
  "Starts a NuPIC model from the model params returned by the swarm\n"
  "and pushes each line of input from the gym into the model. Results\n"
  "are written to an output file (default) or plotted dynamically if\n"
  "the --plot option is specified.\n"
)
# CC_NAME = "credit_card_small"
# CC_NAME = "credit_card_few_features_med"
CC_NAME = "credit_card_all_features_med"
DATA_DIR = "."
MODEL_PARAMS_DIR = "./model_params"
# '2017-08-24 03:19:52'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def createModel(modelParams):
  """
  Given a model params dictionary, create a CLA Model. Automatically enables
  inference for amount.
  :param modelParams: Model params dict
  :return: OPF Model object
  """
  model = ModelFactory.create(modelParams)
  #In case of anomaly detection, this predicted field is somewhat optional
  model.enableInference({"predictedField": "class"})
  return model



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

  counter = 0
  for row in csvReader:
    counter += 1
    if (counter % 100 == 0):
      print "Read %i lines..." % counter
    timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
    # amount = float(row[1])
    v1 = float(row[1])
    v2 = float(row[2])
    v3 = float(row[3])
    v4 = float(row[4])
    v5 = float(row[5])
    v6 = float(row[6])
    v7 = float(row[7])
    v8 = float(row[8])
    v9 = float(row[9])
    v10 = float(row[10])
    v11 = float(row[11])
    v12 = float(row[12])
    v13 = float(row[13])
    v14 = float(row[14])
    v15 = float(row[15])
    v16 = float(row[16])
    v17 = float(row[17])
    v18 = float(row[18])
    v19 = float(row[19])
    v20 = float(row[20])
    v21 = float(row[21])
    v22 = float(row[22])
    v23 = float(row[23])
    v24 = float(row[24])
    v25 = float(row[25])
    v26 = float(row[26])
    v27 = float(row[27])
    v28 = float(row[28])
    amount = float(row[29])
    cls = int(row[30])
    # print("Row %i, amount read %f " %(counter, amount))
    result = model.run({
      "timestamp": timestamp,
      "V1": v1,
      "V2": v2,
      "V3": v3,
      "V4": v4,
      "V5": v5,
      "V6": v6,
      "V7": v7,
      "V8": v8,
      "V9": v9,
      "V10": v10,
      "V11": v11,
      "V12": v12,
      "V13": v13,
      "V14": v14,
      "V15": v15,
      "V16": v16,
      "V17": v17,
      "V18": v18,
      "V19": v19,
      "V20": v20,
      "V21": v21,
      "V22": v22,
      "V23": v23,
      "V24": v24,
      "V25": v25,
      "V26": v26,
      "V27": v27,
      "V28": v28,
      "amount": amount,
      "class": cls
    })

    if plot:
      result = shifter.shift(result)

    prediction = result.inferences["multiStepBestPredictions"][1]
    anomalyScore = result.inferences["anomalyScore"]
    output.write(timestamp, cls, prediction, anomalyScore)
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
