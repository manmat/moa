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

//import com.bigml.histogram.Histogram;
//import com.bigml.histogram.MixedInsertException;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.sketches.quantiles.UpdateDoublesSketch;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.Parallel;
import moa.classifiers.Regressor;
import moa.classifiers.trees.FIMTQR;
import moa.classifiers.trees.FIMTQRSketch;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.*;
import java.util.function.BinaryOperator;

import static moa.classifiers.meta.AdaptiveRandomForest.calculateSubspaceSize;

/**
 * An ensemble of FIMTQR trees that are trained in the bagging manner as described in the ARF paper
 * Will need to pull out histograms from trees, in order to combine and then provide quantile prediction.
 * Usage: EvaluatePrequentialRegression -e (IntervalRegressionPerformanceEvaluator -w 10000) -l (meta.OnlineQRF -t 5 -b 100 -a 90) -s (ArffFileStream -f somefile.arff)  -f 10000
 */
public class OnlineQRFSketch extends AbstractClassifier implements Regressor, Parallel {
  private double quantileUpper;
  private double quantileLower;
  private FIMTQRSketch[] ensemble;
  private int subspaceSize;
  long instancesSeen;

  private ExecutorService executor;
  private CompletionService<UpdateDoublesSketch> ecs;
  private ForkJoinPool forkJoinPool; // Needed because the streams interface will take all processors otherwise

  public ClassOption baseLearnerOption = new ClassOption("treeLearner", 'l',
      "Random Forest Tree.", Classifier.class,"trees.FIMTQRSketch -e");

  public IntOption ensembleSize = new IntOption("ensembleSize", 's', "Number of trees in the ensemble",
      10, 1, Integer.MAX_VALUE);

  public FloatOption confidenceLevel = new FloatOption("confidenceLevel", 'a',
      "The confidence level as a float (e.g. 0.95 = 95% prediction interval)",
      0.9, 0.0, 1.0);

  public IntOption numBins = new IntOption(
      "numBins", 'b', "Number of bins to use at leaf histograms",
      128, 1, Integer.MAX_VALUE);

  public FloatOption lambdaOption = new FloatOption("lambda", 'd',
      "The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);


  public MultiChoiceOption mFeaturesModeOption = new MultiChoiceOption("mFeaturesMode", 'o',
      "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features.",
      new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)",
          "Percentage (M * (m / 100))"},
      new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 3);

  public IntOption mFeaturesPerTreeSizeOption = new IntOption("mFeaturesPerTreeSize", 'm',
      "Number of features allowed considered for each split. Negative values corresponds to M - m", 100, Integer.MIN_VALUE, Integer.MAX_VALUE);

  public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
      "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);


  @Override
  public double[] getVotesForInstance(Instance inst) {
    // TODO: Will prolly need an option to return a single/mean value as well
    if(this.ensemble == null)
      initEnsemble(inst);

    // Gather and merge all histograms
    UpdateDoublesSketch combinedSketch;
    if (executor != null) {
      combinedSketch = multiThreadedPredict(inst);
    } else {
      combinedSketch = singleThreadedPredict(inst);
    }

    // Get quantile from merged histograms
    assert combinedSketch != null;
    if (combinedSketch.isEmpty())
      return new double[]{0.0, 0.0};
    Double quantilePredictionLower = combinedSketch.getQuantile(quantileLower);
    Double quantilePredictionUpper = combinedSketch.getQuantile(quantileUpper);
    return new double[]{quantilePredictionLower, quantilePredictionUpper};
  }

  private UpdateDoublesSketch singleThreadedPredict(Instance inst) {
    com.yahoo.sketches.quantiles.DoublesUnion union = com.yahoo.sketches.quantiles.DoublesUnion.builder().build();
    // We iterate through all learners, and merge histograms as we go
    for (FIMTQRSketch member : ensemble) {
      if (!member.trainingHasStarted()) {
        continue;
      }
      UpdateDoublesSketch curSketch = member.getPredictionSketch(inst);
      //try {
        //This could be done async as well
      union.update(curSketch); // tvas: Modification should happen in place, check!
      //} catch (MixedInsertException e) {
      //  e.printStackTrace();
      //}
    }

    return union.isEmpty() ? UpdateDoublesSketch.builder().setK(numBins.getValue()).build() : union.getResult();
  }

  private UpdateDoublesSketch multiThreadedPredict(Instance inst) {
    ArrayList<Future<UpdateDoublesSketch>> sketchesFutures = new ArrayList<>();
    //com.yahoo.sketches.quantiles.DoublesUnion union = com.yahoo.sketches.quantiles.DoublesUnion.builder().build();
    UpdateDoublesSketch combinedSketch = null;

    for (FIMTQRSketch member : ensemble) {
      if (!member.trainingHasStarted()) {
        return UpdateDoublesSketch.builder().setK(numBins.getValue()).build();
      }
      if (this.executor != null) {
        sketchesFutures.add(ecs.submit(new SketchPredictionRunnable(member, inst)));
      }
    }

    // This will do the predictions in parallel
    ArrayList<UpdateDoublesSketch> sketches = new ArrayList<>(sketchesFutures.size());
    for (Future ignored : sketchesFutures) {
      try {
        UpdateDoublesSketch sketch = ecs.take().get();
//        combinedHist.merge(hist);
        sketches.add(sketch);
      } catch (InterruptedException | ExecutionException e) {
        e.printStackTrace();
      }
    }

    // This operator merges two histograms into a new one
    BinaryOperator<UpdateDoublesSketch> merger = (h1, h2) -> {
      //try {
        // Need to create new object because merge is in-place
        com.yahoo.sketches.quantiles.DoublesUnion ret = com.yahoo.sketches.quantiles.DoublesUnion.builder().build();
        ret.update(h1);
        ret.update(h2);
        return ret.getResult();
      //} catch (MixedInsertException e) {
      //  e.printStackTrace();
      //  return new Histogram(numBins.getValue());
      //}
    };
    // This will do the merging in parallel
    try {
       combinedSketch = forkJoinPool.submit(() -> sketches.parallelStream()
          .reduce(UpdateDoublesSketch.builder().setK(numBins.getValue()).build(), merger)).get();
    } catch (InterruptedException | ExecutionException e) {
      e.printStackTrace();
    }

    return combinedSketch;
  }

  @Override
  public void resetLearningImpl() {
    // Translate confidence to upper and lower quantiles
    double halfSignificance = (1.0 - confidenceLevel.getValue()) / 2.0; // We divide by two for each region (lower,upper)
    quantileLower = 0.0 + halfSignificance;
    quantileUpper = 1.0 - halfSignificance;

    // Multi-threading
    int numberOfJobs;
    if(this.numberOfJobsOption.getValue() == -1)
      numberOfJobs = Runtime.getRuntime().availableProcessors();
    else
      numberOfJobs = this.numberOfJobsOption.getValue();

    if(numberOfJobs != 1) {
      executor = Executors.newFixedThreadPool(numberOfJobs);
      ecs = new ExecutorCompletionService<>(executor);
      forkJoinPool = new ForkJoinPool(numberOfJobsOption.getValue());
    }
  }

  @Override
  public void trainOnInstanceImpl(Instance instance) {
    ++this.instancesSeen;
    if(this.ensemble == null)
      initEnsemble(instance);

    Collection<TrainingRunnable> trainers = new ArrayList<>();
    for (FIMTQRSketch member : ensemble) {
      // Predict and evaluate here? ARF does this, why?
//      double[] prediction = member.getVotesForInstance(instance);
      int k = MiscUtils.poisson(lambdaOption.getValue(), this.classifierRandom);
      if (k > 0) {
        Instance weightedInstance = instance.copy();
        weightedInstance.setWeight(k);
        if(this.executor != null) {
          TrainingRunnable trainer = new TrainingRunnable(member,
              weightedInstance);
          trainers.add(trainer);
        }
        else {
          member.trainOnInstance(weightedInstance);
        }
      }
    }
    // Using invokeAll and Runnables.
    // tvas: There are guarantees that the futures will complete before the function returns,
    // because of the implementation of invokeAll (AbstractExecutorService). It's still sequential though.
    if(this.executor != null) {
      try {
        this.executor.invokeAll(trainers);
      } catch (InterruptedException ex) {
        throw new RuntimeException("Could not call invokeAll() on training threads.");
      }
    }
    // tvas: Using collection service. Seems like this is slower than invokeAll, no idea why, should test more
//    if(this.executor != null) {
//
//      for (TrainingRunnable trainingRunnable : trainers) {
//        ecs.submit(trainingRunnable);
//      }
//      // Ensure all tasks have completed before moving on
//      for (TrainingRunnable ignored : trainers) {
//        try {
//          final Future<Integer> res = ecs.take();
//          Integer j = res.get();
//        } catch (InterruptedException | ExecutionException e) {
//          e.printStackTrace();
//        }
//      }
//    }
  }

  // Mostly copied over from AdaptiveRandomForest
  protected void initEnsemble(Instance instance) {
    // Init the ensemble.
    int ensembleSize = this.ensembleSize.getValue();
    ensemble = new FIMTQRSketch[ensembleSize];

    subspaceSize = calculateSubspaceSize(
        mFeaturesPerTreeSizeOption.getValue(), mFeaturesModeOption.getChosenIndex(), instance);

    FIMTQRSketch baseLearner = (FIMTQRSketch) getPreparedClassOption(this.baseLearnerOption);
    baseLearner.subspaceSizeOption.setValue(subspaceSize);
    baseLearner.numBins.setValue(numBins.getValue());
    baseLearner.resetLearning();

    // TODO: Ended up breaking encapsulation. Need to generalize if we want to use trees other than FIMT
    for (int i = 0; i < ensembleSize; i++) {
      ensemble[i] = (FIMTQRSketch) baseLearner.copy();
      ensemble[i].resetLearning();
      ensemble[i].treeID = i;
    }

  }

  @Override
  protected Measurement[] getModelMeasurementsImpl() {
    return null;
  }

  @Override
  public void getModelDescription(StringBuilder out, int indent) {

  }

  @Override
  public boolean isRandomizable() {
    return true;
  }

  @Override
  public void shutdownExecutor() {
    if (executor != null) {
      executor.shutdown();
      forkJoinPool.shutdown();
    }
  }

  @Override
  public Classifier[] getSubClassifiers() {
    return this.ensemble;
  }

  /*
  class HistogramPredictionRunnable implements Runnable, Callable<Histogram> {
    final private FIMTQR learner;
    final private Instance instance;
    private Histogram histogram;

    public HistogramPredictionRunnable(FIMTQR learner, Instance instance) {
      this.learner = learner;
      this.instance = instance;
    }

    @Override
    public void run() {
      histogram = learner.getPredictionHistogram(this.instance);
    }

    @Override
    public Histogram call() {
      run();
      return histogram;
    }
  }
  */

  class SketchPredictionRunnable implements Runnable, Callable<UpdateDoublesSketch> {
    final private FIMTQRSketch learner;
    final private Instance instance;
    private UpdateDoublesSketch sketch;

    public SketchPredictionRunnable(FIMTQRSketch learner, Instance instance) {
      this.learner = learner;
      this.instance = instance;
    }

    @Override
    public void run() {
      sketch = learner.getPredictionSketch(this.instance);
    }

    @Override
    public UpdateDoublesSketch call() {
      run();
      return sketch;
    }
  }

}
