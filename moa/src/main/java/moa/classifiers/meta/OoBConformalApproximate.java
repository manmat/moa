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

import com.yahoo.labs.samoa.instances.Instance;

import java.lang.reflect.Array;
import java.util.*;

public class OoBConformalApproximate extends OoBConformalRegressor{

  @Override
  protected void updateCalibrationScores() {
    calibrationScores.clear();
    for (Map.Entry<Instance, HashMap<Integer, Double>> instancePredictionsMap : instanceToLearnerToPrediction.entrySet()) {
      Instance curInstance = instancePredictionsMap.getKey();
      HashMap<Integer, Double> predictorIndexPredictionMap = instancePredictionsMap.getValue();
      double sum = 0;
      for (Map.Entry<Integer, Double> predictorIndexPredictionEntry : predictorIndexPredictionMap.entrySet()) {
        sum += predictorIndexPredictionEntry.getValue();
      }
      double prediction = sum / predictorIndexPredictionMap.size();
      double trueValue = curInstance.classValue();
      double calScore = errorFunction(prediction, trueValue);
      calibrationScores.put(curInstance, calScore);
    }
  }
}
