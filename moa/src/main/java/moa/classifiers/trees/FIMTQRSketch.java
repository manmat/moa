package moa.classifiers.trees;
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

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import com.yahoo.sketches.quantiles.DoublesSketch;
import com.yahoo.sketches.quantiles.UpdateDoublesSketch;
import moa.classifiers.core.attributeclassobservers.FIMTDDNumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.core.Measurement;
import moa.core.SizeOf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class FIMTQRSketch extends FIMTDD {

  public IntOption numBins = new IntOption(
      "numBins", 'b', "Number of bins to use at leaf histograms",
      100, 1, Integer.MAX_VALUE);

  public FIMTQRSketch() {

  }

  public FIMTQRSketch(int numBins) {
    this.numBins.setValue(numBins);
  }

  public FIMTQRSketch(int numBins, int subspaceSize) {
    this.numBins.setValue(numBins);
    this.subspaceSizeOption.setValue(subspaceSize);
  }

  // Thin interface to define common function for leaves
  // Idea stolen from Scala traits and here: https://stackoverflow.com/a/21824485/209882
  // tvas: We could generalize this and allow any tree regressor do this type of learning, I think.
  public interface withSketch {
    UpdateDoublesSketch getPredictionSketch(Instance instance);
  }

  public static class QRInactiveNode extends InactiveLearningNode implements withSketch {
    protected UpdateDoublesSketch labelSketch;

    public QRInactiveNode(LeafNode leafNode) {
      super(leafNode);
      assert leafNode instanceof withSketch;
      labelSketch = ((QRLeafNode) leafNode).getLabelSketch();
    }

    @Override
    public void learnFromInstance(Instance inst) {
      // Update the statistics for this node
      // number of instances passing through the node
      examplesSeen += inst.weight();
      // sum of y values
      sumOfValues += inst.weight() * inst.classValue();
      // sum of squared y values
      sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();
      // sum of absolute errors
      sumOfAbsErrors += inst.weight() * Math.abs(tree.normalizeTargetValue(Math.abs(inst.classValue() - getPrediction(inst))));
      //try {
      labelSketch.update(inst.classValue());
      //} catch (MixedInsertException e) {
      //  e.printStackTrace();
      //}
    }

    @Override
    public UpdateDoublesSketch getPredictionSketch(Instance instance) {
      return labelSketch;
    }

  }

  public static class QRLeafNode extends LeafNode implements withSketch{

    private UpdateDoublesSketch labelSketch;

    private int[] attributeIndexList;

    private int subspaceSize;

    /**
     * Create a new LeafNode
     *
     * @param tree
     */
    public QRLeafNode(FIMTQRSketch tree) {
      super(tree);
      labelSketch = DoublesSketch.builder().build(); //new Histogram(tree.numBins.getValue());
      subspaceSize = tree.subspaceSizeOption.getValue();
    }

    public QRLeafNode(FIMTQRSketch tree, Node existingNode) {
      super(tree, existingNode);
      assert existingNode instanceof QRInactiveNode;
      // We fake an instance because we are sure the node should be a leaf node if it was previously deactivated
      Instance dummy = new InstanceImpl(0);
      labelSketch = ((withSketch) existingNode).getPredictionSketch(dummy);
      subspaceSize = tree.subspaceSizeOption.getValue();
    }

    @Override
    public void learnFromInstance(Instance inst) {
      // Create a list of unique attribute indices with subspaceSize elements
      // tvas: This seems more reasonable than what ARFHoeffdingTree does, shouldn't make
      // much diff in performance anyway, but could try micro-benching if it turns out this matters.
      if (attributeIndexList == null) {
        // Create a list with numbers 0-numAttributes-1
        attributeIndexList = new int[subspaceSize];
        ArrayList<Integer> list = new ArrayList<>();
        for (int i=0; i < inst.numAttributes() - 1; i++) { // Minus one here because numAttributes includes class
          list.add(i);
        }
        // Shuffle the list
        // tvas: Assuming every element has equal chance to be in first subspaceSize elements
        Collections.shuffle(list);
        // Select the first subspaceSize elements to be the feature ids for this
        for (int i = 0; i < subspaceSize; i++) {
          attributeIndexList[i] = list.get(i);
        }
      }
      // Do the learning only on selected attributes (duplicating FIMTDD code)

      // Update the statistics for this node
      // number of instances passing through the node
      examplesSeen += inst.weight();
      // sum of y values
      sumOfValues += inst.weight() * inst.classValue();
      // sum of squared y values
      sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();
      // sum of absolute errors
      sumOfAbsErrors += inst.weight() * Math.abs(tree.normalizeTargetValue(Math.abs(inst.classValue() - getPrediction(inst))));

      if (tree.buildingModelTree()) learningModel.updatePerceptron(inst);

      for (int j = 0; j < subspaceSize - 1; j++) {
        int i = attributeIndexList[j];
        int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
        FIMTDDNumericAttributeClassObserver obs = attributeObservers.get(i);
        if (obs == null) {
          // At this stage all nominal attributes are ignored
          if (inst.attribute(instAttIndex).isNumeric()) {
            obs = tree.newNumericClassObserver();
            this.attributeObservers.set(i, obs);
          }
        }
        if (obs != null) {
          obs.observeAttributeClass(inst.value(instAttIndex), inst.classValue(), inst.weight());
        }
      }

      if (tree.growthAllowed) {
        checkForSplit(tree);
      }
      //try {
      labelSketch.update(inst.classValue());
      //} catch (MixedInsertException e) {
      //  e.printStackTrace();
      //}
    }

    @Override
    public UpdateDoublesSketch getPredictionSketch(Instance instance) {
      return labelSketch;
    }

    public UpdateDoublesSketch getLabelSketch() {
      return labelSketch;
    }
  }

  public static class QRSPlitNode extends SplitNode implements withSketch{

    /**
     * Create a new SplitNode
     *
     * @param splitTest
     * @param tree
     */
    public QRSPlitNode(InstanceConditionalTest splitTest, FIMTDD tree) {
      super(splitTest, tree);
    }

    public UpdateDoublesSketch getPredictionSketch(Instance instance) {
      Node curNode = children.get(splitTest.branchForInstance(instance));
      assert curNode instanceof withSketch : "Incorrect QRSplitNode class: " + curNode.getClass();
      return ((withSketch) curNode).getPredictionSketch(instance);
    }
  }


  @Override
  protected LeafNode newLeafNode() {
    maxID++;
    return new QRLeafNode(this);
  }

  @Override
  protected LeafNode newLeafNode(Node existingNode) {
    return new QRLeafNode(this, existingNode);
  }

  @Override
  protected SplitNode newSplitNode(InstanceConditionalTest splitTest) {
    maxID++;
    return new QRSPlitNode(splitTest, this);
  }

  @Override
  protected InactiveLearningNode createInactiveNode(LeafNode existingNode) {
    return new QRInactiveNode(existingNode);
  }

  public UpdateDoublesSketch getPredictionSketch(Instance instance) {
    return ((withSketch)treeRoot).getPredictionSketch(instance);
  }

  private int measureHistogramSize() {
    FoundNode[] learningNodes = findLearningNodes();
    int sizeSum = 0;
    Instance dummy = new InstanceImpl(0);
    for (FoundNode learningNode : learningNodes) {
      UpdateDoublesSketch labelHistogram = ((withSketch) learningNode.node).getPredictionSketch(dummy);
      sizeSum += SizeOf.fullSizeOf(labelHistogram);
    }

    return sizeSum;
  }

  @Override
  protected Measurement[] getModelMeasurementsImpl() {
    Measurement[] fimtddMetrics =  super.getModelMeasurementsImpl();
    Measurement histogramSize = new Measurement("histogram size (bytes)",
        measureHistogramSize());
    Measurement[] newMetrics = Arrays.copyOf(fimtddMetrics, fimtddMetrics.length + 1);
    newMetrics[fimtddMetrics.length] = histogramSize;
    return newMetrics;
  }
}
