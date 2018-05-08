package moa.classifiers.meta;
/*
 * #%L
 * SAMOA
 * %%
 * Copyright (C) 2014 - 2015 Apache Software Foundation
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Measurement;

import java.lang.reflect.Array;
import java.util.*;

public class OoBConformalApproximate extends OoBConformalRegressor{

  public FloatOption elementRecalibrationRation = new FloatOption("elementRecalibrationRation", 'e',
          "The ration of the changed learners to all oob learners to recalculate the calibration score for a given instance", 0.5, 0.0, 1.0);

  public FloatOption setRecalibrationRation = new FloatOption("setRecalibrationRation", 'r',
          "The ration of the instances that need recalculation to all recalibration instances at which we will recalculate the scores", 0.5, 0.0, 1.0);


  @Override
  protected void updateCalibrationScores() {
    calibrationScores.clear();
    HashSet<Instance> instancesToRecalculate = new HashSet<>();
    for (Map.Entry<Instance, HashMap<Integer, Double>> instancePredictionsMap : instanceToLearnerToPrediction.entrySet()) {
      Instance curInstance = instancePredictionsMap.getKey();
      HashMap<Integer, Double> predictorIndexPredictionMap = instancePredictionsMap.getValue();
      int oobPredictors = predictorIndexPredictionMap.size();
      int needsRecalc = 0;
      long instanceWasRecalculatedLast = timeOfLastRecalculation.get(curInstance);
      for (Map.Entry<Integer, Double> predictorIndexPredictionEntry : predictorIndexPredictionMap.entrySet()) {
        Integer ensembleIndex = predictorIndexPredictionEntry.getKey();
        if (timeOfUpdate[ensembleIndex] > instanceWasRecalculatedLast)
          needsRecalc++;
      }
      double ratio = needsRecalc / (double) oobPredictors;
      assert ratio <= 1.0 : "Error ratio was: " + ratio;
      if (ratio >= this.elementRecalibrationRation.getValue()){
        instancesToRecalculate.add(curInstance);
      }
    }
    boolean doRecalculation = instancesToRecalculate.size() / (float) instanceToLearnerToPrediction.size() >= this.setRecalibrationRation.getValue();
    for (Map.Entry<Instance, HashMap<Integer, Double>> instancePredictionsMap : instanceToLearnerToPrediction.entrySet()) {
      Instance curInstance = instancePredictionsMap.getKey();
      HashMap<Integer, Double> newPredictions = new HashMap<>();
      HashMap<Integer, Double> predictorIndexPredictionMap = instancePredictionsMap.getValue();
      long instanceWasRecalculatedLast = timeOfLastRecalculation.get(curInstance);
      if (doRecalculation && instancesToRecalculate.contains(curInstance)){
        for (Map.Entry<Integer, Double> predictorIndexPredictionEntry : predictorIndexPredictionMap.entrySet()){
          Integer ensembleIndex = predictorIndexPredictionEntry.getKey();
          if (timeOfUpdate[ensembleIndex] > instanceWasRecalculatedLast){
            double pred = ensemble[ensembleIndex].getVotesForInstance(curInstance)[0];
            numOfVoteCalls++;
            newPredictions.put(ensembleIndex, pred);
          } else {
            newPredictions.put(ensembleIndex, predictorIndexPredictionEntry.getValue());
          }
        }
        timeOfLastRecalculation.put(curInstance, trainingInstances);
      } else {
        newPredictions = predictorIndexPredictionMap;
      }
      double sum = 0;
      for (double val: newPredictions.values()){
        sum += val;
      }
      double prediction = sum / predictorIndexPredictionMap.size();
      double trueValue = curInstance.classValue();
      calibrationScores.put(curInstance, errorFunction(prediction, trueValue));
    }
    assert (calibrationScores.size() == instanceToLearnerToPrediction.size());
  }

  @Override
  protected Measurement[] getModelMeasurementsImpl() {
    return new Measurement[]{
            new Measurement("average training time (nanosec)", trainingTime / (double) trainingInstances),
            new Measurement("number of getVotes calls", numOfVoteCalls)
    };
  }
}
