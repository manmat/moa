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
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.Parallel;
import moa.classifiers.trees.FIMTDD;
import moa.classifiers.trees.FIMTQR;
import moa.core.MiscUtils;

import java.util.*;
import java.util.concurrent.*;

import org.apache.commons.math3.distribution.NormalDistribution;

import static moa.classifiers.meta.AdaptiveRandomForest.calculateSubspaceSize;

// There ought to be an abstract class for CP that implements the common stuff (like the arraylist for scores)
public class OoBConformalRegressor extends ConformalRegressor implements Parallel {
  protected double quantileUpper;
  protected double quantileLower;

  protected FIMTDD[] ensemble;
  private boolean[] wasUpdatedLast;
  protected int subspaceSize;
  protected int maxCalibrationInstances;

  protected HashMap<Instance, HashMap<Integer, Double>> instanceToLearnerToPrediction;

  protected ExecutorService executor;
  protected CompletionService<double[]> ecs;

  public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
      "The number of trees in the ensemble.", 10, 1, Integer.MAX_VALUE);

  public MultiChoiceOption mFeaturesModeOption = new MultiChoiceOption("mFeaturesMode", 'o',
      "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features.",
      new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)",
          "Percentage (M * (m / 100))"},
      new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 1);

  public IntOption mFeaturesPerTreeSizeOption = new IntOption("mFeaturesPerTreeSize", 'm',
      "Number of features allowed considered for each split. Negative values corresponds to M - m", 2, Integer.MIN_VALUE, Integer.MAX_VALUE);

  public FloatOption lambdaOption = new FloatOption("lambda", 'd',
      "The lambda parameter for bagging.", 1.0, 1.0, Float.MAX_VALUE);

  public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
      "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);

  @Override
  public void trainOnInstanceImpl(Instance inst) {

    HashMap<Integer, Double> oobTreeIndicesToPredictions = commonTraining(inst);
    // TODO: Have a "burn-in" period for the algo, where we ensure the first x
    // data points end up as OoB for at least one learner. That way we fill up
    // the calibration set as soon as possible
    if (!oobTreeIndicesToPredictions.isEmpty()) {
      if (instanceToLearnerToPrediction.size() < maxCalibrationInstances) {
        instanceToLearnerToPrediction.put(inst, oobTreeIndicesToPredictions);
      } else { // this way we are removing a random (?) element from the map
        Set<Instance> instanceSet = instanceToLearnerToPrediction.keySet();
        Instance[] instances = instanceSet.toArray(new Instance[instanceSet.size()]);

        instanceToLearnerToPrediction.remove(instances[0]);
        instanceToLearnerToPrediction.put(inst, oobTreeIndicesToPredictions);
      }
    }

    updateCalibrationScores();
  }

  protected HashMap<Integer, Double> commonTraining(Instance inst) {
    Arrays.fill(wasUpdatedLast, false);
    HashMap<Integer, Double> oobTreeIndicesToPredictions = new HashMap<>();

    // tvas: Alternative is to have a map from instance to a tuple (predictorIndexList, predictionList)
    // That way I have access to the indices of the predictors and their corresponding predictions in a
    // perhaps easier to iterate format. Guava matrix works as well here.
    if (this.ensemble == null)
      initEnsemble(inst);
    Collection<TrainingRunnable> inBag = new ArrayList<>();
    Collection<OoBPredictionRunnable> outOfBag = new ArrayList<>();
    for (int i = 0; i < ensemble.length; i++) {
      int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
      if (k > 0) {
        wasUpdatedLast[i] = true;
        Instance weightedInstance = inst.copy();
        weightedInstance.setWeight(k);
        if(this.executor != null) {
          TrainingRunnable trainer = new TrainingRunnable(ensemble[i],
              weightedInstance);
          inBag.add(trainer);
        }
        else {
          ensemble[i].trainOnInstance(weightedInstance);
        }
      } else {
        if (ensemble[i].trainingHasStarted()) {
          if (this.executor != null) {
            OoBPredictionRunnable predictor = new OoBPredictionRunnable(ensemble[i], inst, i);
            outOfBag.add(predictor);
          } else {
            double[] curPred =  ensemble[i].getVotesForInstance(inst);
            assert curPred.length == 1;
            oobTreeIndicesToPredictions.put(i, curPred[0]);
          }
        }
      }
    }

    List<Future<AbstractMap.Entry>> oobPredictionFutures;
    if(executor != null) {
      try {
        executor.invokeAll(inBag);
        oobPredictionFutures = executor.invokeAll(outOfBag);
      } catch (InterruptedException ex) {
        throw new RuntimeException("Could not call invokeAll() on training threads.");
      }

      // TODO: The rest of the code could also go to the beginning of the getVotesForInstance function
      // (we just need to make oobPredictionFutures to be a class attribute.
      // That way the oob predictions can happen async until they are needed to make the next interval prediction.
      // TODO: IMPORTANT: For that to work, we can't use invokeAll above, because it blocks until futures are ready
      // TODO: Collection service would be better here
      for (Future<AbstractMap.Entry> future : oobPredictionFutures) {
        try {
          AbstractMap.Entry treeIndexToPred = future.get();
          oobTreeIndicesToPredictions.put((Integer) treeIndexToPred.getKey(), (Double) treeIndexToPred.getValue());
        } catch (InterruptedException | ExecutionException e) {
          e.printStackTrace();
        }
      }
    }

    return oobTreeIndicesToPredictions;
  }

  @Override
  protected void updateCalibrationScores() {
    // TODO: Optimize, not necessary to update all scores with every tree update
    double[] predictions = new double[instanceToLearnerToPrediction.size()];
    double[] trueValues = new double[instanceToLearnerToPrediction.size()];
    int i = 0; // Used to keep track of which calibration instance we are checking
    for (Map.Entry<Instance, HashMap<Integer, Double>> instancePredictionsMap : instanceToLearnerToPrediction.entrySet()) {
      Instance curInstance = instancePredictionsMap.getKey();
      HashMap<Integer, Double> predictorIndexPredictionMap = instancePredictionsMap.getValue();
      double sum = 0;
      for (Map.Entry<Integer, Double> predictorIndexPredictionEntry : predictorIndexPredictionMap.entrySet()) {
        int ensembleIndex = predictorIndexPredictionEntry.getKey();
        double pred;
        if (wasUpdatedLast[ensembleIndex]) {
          pred = ensemble[predictorIndexPredictionEntry.getKey()].getVotesForInstance(curInstance)[0];
          predictorIndexPredictionEntry.setValue(pred);
        } else {
          pred = predictorIndexPredictionEntry.getValue();
        }
        sum += pred;
      }
      double prediction = sum / predictorIndexPredictionMap.size();
      predictions[i] = prediction;
      trueValues[i] = curInstance.classValue();
      i++;
    }

    double[] calScores = errorFunction(predictions, trueValues);
    // TODO: For the approximate version of the algorithm, find which sorting algo is best
    // for almost sorted lists (binary search then simple insertion?)
    Arrays.sort(calScores);
    calibrationScores = calScores;
  }

  @Override
  public double[] getVotesForInstance(Instance inst) {
    MomentAggregate curAggegate = getMoments(inst);
    if (calibrationScores.length < 2) {
      // TODO: Predictive sd is a bad estimate, come up with something else
      // tvas: Depending on the lambda setting, it could be a while until we get 10 cal instances, careful!
      // One option: https://stats.stackexchange.com/a/255131/16052
      return calculateGaussianInterval(curAggegate.mean, Math.sqrt(curAggegate.variance));
    }
    double interval = inverseErrorFunction(confidenceOption.getValue());
    return new double[]{curAggegate.mean - interval, curAggegate.mean + interval};
  }

  protected MomentAggregate getMoments(Instance inst) {
    if(this.ensemble == null)
      initEnsemble(inst);
    MomentAggregate curAggegate = new MomentAggregate(0, 0, 0);
    ArrayList<Future<double[]>> voteFutures = new ArrayList<>();
    for (Classifier member : ensemble) {
      if (executor != null) {
        voteFutures.add(ecs.submit(new PredictionRunnable(member, inst)));
      } else {
        double[] curVotes = member.getVotesForInstance(inst);
        curAggegate = updateMoments(curAggegate, curVotes[0]);
      }
    }
    if (executor != null) {
      for (Future ignored : voteFutures) {
        try {
          double[] curVotes = ecs.take().get();
          // TODO: Can probably have the moment update be done in a parallel reduce as well
          curAggegate = updateMoments(curAggegate, curVotes[0]);
        } catch (InterruptedException | ExecutionException e) {
          e.printStackTrace();
        }
      }
    }
    return finalizeMoments(curAggegate);
  }


  protected double[] calculateGaussianInterval(double mean, double std) {
    if (std < 1e-6) {
      std = 1e-8;
    }
    NormalDistribution norm = new NormalDistribution(mean, std);
    double upperValue = norm.inverseCumulativeProbability(quantileUpper);
    double lowerValue = norm.inverseCumulativeProbability(quantileLower);
    if (Double.isNaN(lowerValue)) {
      lowerValue = mean;
    }
    if (Double.isNaN(upperValue)) {
      upperValue = mean;
    }
    assert lowerValue <= upperValue;

    return new double[]{lowerValue, upperValue};
  }

  /**
   * Class used to hold results of an online mean and variance calculation
   */
  public static class MomentAggregate {
    public long count;
    public double mean;
    public double variance;

    public MomentAggregate(long count, double mean, double variance) {
      this.count = count;
      this.mean = mean;
      this.variance = variance;
    }
  }

  /**
   * Will update an existing moment aggregate
   * Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
   * @param existingAggregate An existing aggregate of moment statistics
   * @param newValue The new value to be incorporated
   * @return An updated version of the moment aggregate. Call {finalizeMoments} on this object at the end of the stream.
   */
  public static MomentAggregate updateMoments(MomentAggregate existingAggregate, double newValue) {
    long count = existingAggregate.count;
    double mean = existingAggregate.mean;
    double variance = existingAggregate.variance;
    count++;
    double delta = newValue - mean;
    mean = mean + delta / count;
    double delta2 = newValue - mean;
    variance = variance + delta * delta2;

    return new MomentAggregate(count, mean, variance);
  }

  /**
   * Returns the finalized aggregate moments, after adjusting the sample variance
   * @param existingAggregate A moments aggregate that has all the required samples integrated.
   * @return A {@link MomentAggregate} instance with final mean and variance.
   * Variance will be NaN if less than two samples were used for the aggregate.
   */
  public static MomentAggregate finalizeMoments(MomentAggregate existingAggregate) {
    long count = existingAggregate.count;
    double mean = existingAggregate.mean;
    double variance = existingAggregate.variance;
    variance = variance / (count - 1);

    if (count < 2) {
      return new MomentAggregate(count, mean, Double.NaN);
    } else {
      return new MomentAggregate(count, mean, variance);
    }
  }

  @Override
  public void resetLearningImpl() {
    // Translate confidence to upper and lower quantiles
    super.resetLearningImpl();
    double halfConfidence = (1.0 - confidenceOption.getValue()) / 2.0; // We divide by two for each region (lower,upper)
    quantileLower = 0.0 + halfConfidence;
    quantileUpper = 1.0 - halfConfidence;
    ensemble = null;
    wasUpdatedLast = new boolean[ensembleSizeOption.getValue()];
    instanceToLearnerToPrediction = new HashMap<>();
    maxCalibrationInstances = maxCalibrationInstancesOption.getValue();

    if (!calibrationDataset.getValue().equals("")) {
      System.out.println("WARNING: OoBCOnformalRegression should not take a calibration set! (-c option)");
    }
  }

  protected void initEnsemble(Instance instance) {
    // Init the ensemble.
    int ensembleSize = ensembleSizeOption.getValue();
    ensemble = new FIMTDD[ensembleSize];

    subspaceSize = calculateSubspaceSize(
        mFeaturesPerTreeSizeOption.getValue(), mFeaturesModeOption.getChosenIndex(), instance);

    FIMTDD baseLearner = (FIMTDD) getPreparedClassOption(this.baseLearnerOption);
    baseLearner.subspaceSizeOption.setValue(subspaceSize);
    baseLearner.resetLearning();

    for (int i = 0; i < ensembleSize; i++) {
      ensemble[i] = (FIMTDD) baseLearner.copy();
      ensemble[i].resetLearning();
      ensemble[i].treeID = i;
    }


    // Multi-threading
    int numberOfJobs;
    if(this.numberOfJobsOption.getValue() == -1)
      numberOfJobs = Runtime.getRuntime().availableProcessors();
    else
      numberOfJobs = this.numberOfJobsOption.getValue();

    if(numberOfJobs != 1) {
      executor = Executors.newFixedThreadPool(numberOfJobs);
      ecs = new ExecutorCompletionService<>(executor);
    }
  }

  @Override
  public void shutdownExecutor() {
    if (executor != null) {
      executor.shutdown();
    }
  }

  @Override
  public Classifier[] getSubClassifiers() {
    return this.ensemble;
  }

  class OoBPredictionRunnable implements Runnable, Callable<Map.Entry> {
    final private Classifier learner;
    final private Instance instance;
    private Double vote;
    int learnerIndex;

    public OoBPredictionRunnable(Classifier learner, Instance instance, int learnerIndex) {
      this.learner = learner;
      this.instance = instance;
      this.learnerIndex = learnerIndex;
    }

    @Override
    public void run() {
      double[] votes = learner.getVotesForInstance(this.instance);
      assert votes.length == 1;
      vote = votes[0];
    }

    @Override
    public Map.Entry call() {
      run();
      return new AbstractMap.SimpleEntry<>(learnerIndex, vote);
    }

  }
}
