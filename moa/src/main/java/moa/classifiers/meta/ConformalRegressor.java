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
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.streams.ArffFileStream;

import java.util.ArrayList;
import java.lang.Math;
import java.util.Arrays;

public class ConformalRegressor extends AbstractClassifier implements Regressor{
  public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
      "Regressor to train.", Classifier.class, "trees.FIMTDD -e");

  public StringOption calibrationDataset = new StringOption("calibrationData", 'c',
      "Filepath to the arff file to be used as a calibration dataset", "");

  public FloatOption confidenceOption = new FloatOption("confidenceOption", 'a',
      "Prediction confidenceOption level. Set to 0.9 for a 90% prediction interval.", 0.9);

  public IntOption maxCalibrationInstancesOption = new IntOption("maxCalibrationInstances", 'i',
      "Maximum number of instances to use for calibration", 100, 10, Integer.MAX_VALUE);

  private Classifier model;

  private ArrayList<Instance> calibrationSet;

  double[] calibrationScores;

  private void readCalibrationSet() {
    ArffFileStream stream = new ArffFileStream(calibrationDataset.getValue(), -1);
    int i = 0;
    while (stream.hasMoreInstances() && i < maxCalibrationInstancesOption.getValue()) {
      calibrationSet.add(stream.nextInstance().instance);
      i++;
    }
  }

  double[] errorFunction(double[] predictions, double[] trueValues) {
    // Implementing the absolute error function for now, this could be generalized
    double[] scores = new double[trueValues.length];
    // TODO: Should be done in jBLAS or something
    for (int i = 0; i < trueValues.length; i++) {
      scores[i] = Math.abs(predictions[i] - trueValues[i]);
    }

    return scores;
  }

  /**
   * Inverse of non-conformity function, i.e. calculates prediction interval (PI).
   * @param significance Interval confidenceOption. Example: If we want 90% PIs this should be 0.1
   * @return The value to add/subtract on each side of the prediction to get the PI
   */
  protected double inverseErrorFunction(double significance) {
    // tvas: Maybe this should be a class parameter to avoid re-calculation?
    // tvas: We assume the calibration scores are up-to-date and sorted
//    double[] calibrationScores = reverseArray(this.calibrationScores); // TODO: Not necessary if I just pass confidenceOption as significance?
//    assert isSorted(calibrationScores); // TODO: Debug purposes, remove for testing
    int border = (int) Math.floor(significance * (calibrationScores.length + 1)) - 1;
    border = Math.min(Math.max(border, 0), calibrationScores.length - 1);
    return calibrationScores[border];
  }


  public static boolean isSorted(Double[] data){
    for(int i = 1; i < data.length; i++){
      if(data[i-1] > data[i]){
        return false;
      }
    }
    return true;
  }

  protected void updateCalibrationScores() {
    double[] predictions = new double[calibrationSet.size()];
    double[] trueValues = new double[calibrationSet.size()];
    // tvas: mah performance!
    for (int i = 0; i < predictions.length; i++) {
      Instance calInstance = calibrationSet.get(i);
      trueValues[i] = calInstance.classValue();
      double[] prediction = model.getVotesForInstance(calInstance);
      assert prediction.length == 1;
      predictions[i] = prediction[0];
    }
    double[] calScores = errorFunction(predictions, trueValues);
    Arrays.sort(calScores);
    calibrationScores = calScores;
  }

  @Override
  public double[] getVotesForInstance(Instance inst) {
    if (!model.trainingHasStarted()) {
      return new double[] {0, 0};
    }
    double[] modelPredictionArray = model.getVotesForInstance(inst);
    assert modelPredictionArray.length == 1;
    double modelPrediction = modelPredictionArray[0];
//    double interval = inverseErrorFunction(1.0 - confidenceOption.getValue());
    double interval = inverseErrorFunction(confidenceOption.getValue());
    // TODO: Normalized prediction intervals
    return new double[]{modelPrediction - interval, modelPrediction + interval};
  }

  @Override
  public void resetLearningImpl() {
    model = (Classifier) getPreparedClassOption(baseLearnerOption);
    assert model instanceof Regressor; // Will this work?
    calibrationSet = new ArrayList<>();
    calibrationScores = new double[maxCalibrationInstancesOption.getValue()];
    Arrays.fill(calibrationScores, 0d);

    model.resetLearning();
    if (calibrationDataset.getValue().endsWith(".arff")) {
      readCalibrationSet();
    }
  }

  @Override
  public void trainOnInstanceImpl(Instance inst) {
    model.trainOnInstance(inst);
    updateCalibrationScores(); // tvas: This could happen async as well, just needs to complete before next prediction
  }

  @Override
  protected Measurement[] getModelMeasurementsImpl() {
    return new Measurement[0];
  }

  @Override
  public void getModelDescription(StringBuilder out, int indent) {

  }

  @Override
  public boolean isRandomizable() {
    return true;
  }



}
