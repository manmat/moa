/*
 *    FIMTDD.java
 *    Copyright (C) 2015 Jožef Stefan Institute, Ljubljana, Slovenia
 *    @author Aljaž Osojnik <aljaz.osojnik@ijs.si>
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *    
 *    
 */

package moa.classifiers.trees;

import java.io.Serializable;
import java.util.*;

import com.yahoo.labs.samoa.instances.Instance;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import moa.options.ClassOption;
import moa.AbstractMOAObject;
import moa.classifiers.Regressor;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.FIMTDDNumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.AbstractClassifier;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.SizeOf;
import moa.core.StringUtils;

/*
 * Implementation of FIMTDD, regression and model trees for data streams.
 */

public class FIMTDD extends AbstractClassifier implements Regressor {

	private static final long serialVersionUID = 1L;

	protected Node treeRoot;

	public int treeID = 0;

	protected int leafNodeCount = 0;
	protected int splitNodeCount = 0;
	protected int leafNumAtDeactivate = 0;

	protected double weightSeen = 0.0;
	protected long instancesSeen = 0;
	protected double sumOfValues = 0.0;
	protected double sumOfSquares = 0.0;

	protected DoubleVector sumOfAttrValues = new DoubleVector();
	protected DoubleVector sumOfAttrSquares = new DoubleVector();

	public int maxID = 0;

	protected int decisionNodeCount;

	protected int activeLeafNodeCount;

	protected int inactiveLeafNodeCount;

	protected double inactiveLeafByteSizeEstimate;

	protected double activeLeafByteSizeEstimate;

	protected double byteSizeEstimateOverheadFraction;

	protected boolean growthAllowed = true;

	public FIMTDD() {
	}

	public FIMTDD(int subspaceSize) {
		subspaceSizeOption.setValue(subspaceSize);
	}

	//region ================ OPTIONS ================

	public ClassOption splitCriterionOption = new ClassOption(
			"splitCriterion", 's', "Split criterion to use.",
			SplitCriterion.class, "moa.classifiers.core.splitcriteria.VarianceReductionSplitCriterion");

	public IntOption gracePeriodOption = new IntOption(
			"gracePeriod", 'g', "Number of instances a leaf should observe between split attempts.",
			200, 0, Integer.MAX_VALUE);

	public FloatOption splitConfidenceOption = new FloatOption(
			"splitConfidence", 'c', "Allowed error in split decision, values close to 0 will take long to decide.",
			0.0000001, 0.0, 1.0);

	public FloatOption tieThresholdOption = new FloatOption(
			"tieThreshold", 't', "Threshold below which a split will be forced to break ties.",
			0.05, 0.0, 1.0);

	public FloatOption PageHinckleyAlphaOption = new FloatOption(
			"PageHinckleyAlpha", 'a', "Alpha value to use in the Page Hinckley change detection tests.",
			0.005, 0.0, 1.0);

	public IntOption PageHinckleyThresholdOption = new IntOption(
			"PageHinckleyThreshold", 'h', "Threshold value used in the Page Hinckley change detection tests.",
			50, 0, Integer.MAX_VALUE);

	public FloatOption alternateTreeFadingFactorOption = new FloatOption(
			"alternateTreeFadingFactor", 'f', "Fading factor used to decide if an alternate tree should replace an original.",
			0.995, 0.0, 1.0);

	public IntOption alternateTreeTMinOption = new IntOption(
			"alternateTreeTMin", 'y', "Tmin value used to decide if an alternate tree should replace an original.",
			150, 0, Integer.MAX_VALUE);

	public IntOption alternateTreeTimeOption = new IntOption(
			"alternateTreeTime", 'u', "The number of instances used to decide if an alternate tree should be discarded.",
			1500, 0, Integer.MAX_VALUE);

	public FlagOption regressionTreeOption = new FlagOption(
			"regressionTree", 'e', "Build a regression tree instead of a model tree.");

	public FloatOption learningRatioOption = new FloatOption(
			"learningRatio", 'l', "Learning ratio to used for training the Perceptrons in the leaves.",
			0.02, 0, 1.00);

	public FloatOption learningRateDecayFactorOption = new FloatOption(
			"learningRatioDecayFactor", 'd', "Learning rate decay factor (not used when learning rate is constant).",
			0.001, 0, 1.00);

	public FlagOption learningRatioConstOption = new FlagOption(
			"learningRatioConst", 'q', "Keep learning rate constant instead of decaying.");

	public FlagOption prePruneOption = new FlagOption("prePrune", 'p',
			"Enable pre-pruning.");

	public FlagOption disableMemoryManagement = new FlagOption("disableMemoryManagement", 'v',
			"Disable memory management.");

	public IntOption maxByteSizeOption = new IntOption("maxByteSize", 'm',
			"Maximum memory consumed by the tree.", Integer.MAX_VALUE, 0,
			Integer.MAX_VALUE);

	public FlagOption stopMemManagementOption = new FlagOption(
			"stopMemManagement", 'z',
			"Stop growing as soon as memory limit is hit.");

	public IntOption memoryEstimatePeriodOption = new IntOption(
			"memoryEstimatePeriod", 'x',
			"How many instances between memory consumption checks.", 1000000,
			0, Integer.MAX_VALUE);

	public IntOption subspaceSizeOption = new IntOption("subspaceSizeSize", 'k',
			"Number of features per subset for each node split. Negative values = #features - k",
			100, Integer.MIN_VALUE, Integer.MAX_VALUE);


	//endregion ================ OPTIONS ================

	//region ================ CLASSES ================

	public abstract static class Node extends AbstractMOAObject {

		private static final long serialVersionUID = 1L;

		public int ID;

		protected FIMTDD tree;

		protected boolean changeDetection = false;

		protected Node parent;

		protected Node alternateTree;
		protected Node originalNode;

		// The statistics for this node:
		// Number of instances that have reached it
		protected double examplesSeen;
		// Sum of y values
		protected double sumOfValues;
		// Sum of squared y values
		protected double sumOfSquares;
		// Sum of absolute errors
		protected double sumOfAbsErrors; // Needed for PH tracking of mean error

		public Node(FIMTDD tree) {
			this.tree = tree;
			ID = tree.maxID;
		}

		public void copyStatistics(Node node) {
			examplesSeen = node.examplesSeen;
			sumOfValues = node.sumOfValues;
			sumOfSquares = node.sumOfSquares;
			sumOfAbsErrors = node.sumOfAbsErrors;
		}

		public int calcByteSize() {
			return (int) (SizeOf.fullSizeOf(this));
		}

		public int calcByteSizeIncludingSubtree() {
			return calcByteSize();
		}

		/**
		 * The promise of a leaf for splitting is correlated with number of examples that reach it,
		 * in combination with a high number of mistakes.
		 * We take this approach from the HoeffdingTree implementation to use the sum of absolute
		 * errors in a leaf as its promise.
		 *
		 * @return A number representing how promising this leaf is if we were to split it, higher is better.
		 */
		public double calculatePromise() {
			return sumOfAbsErrors;
		}

		/**
		 * Set the parent node
		 */
		public void setParent(Node parent) {
			this.parent = parent;
		}

		/**
		 * Return the parent node
		 */
		public Node getParent() {
			return parent;
		}

		public void disableChangeDetection() {
			changeDetection = false;
		}

		public void restartChangeDetection() {
			changeDetection = true;
		}

		public void getDescription(StringBuilder sb, int indent) {

		}

		public double getPrediction(Instance inst) {
			return 0;
		}

		public void describeSubtree(StringBuilder out, int indent) {
			StringUtils.appendIndented(out, indent, "Leaf");
		}
//
//		public int getLevel() {
//			Node target = this;
//			int level = 0;
//			while (target.getParent() != null) {
//				if (target.skipInLevelCount()) {
//					target = target.getParent();
//					continue;
//				}
//				level = level + 1;
//				target = target.getParent();
//			}
//			if (target.originalNode == null) {
//				return level;
//			} else {
//				return level + originalNode.getLevel();
//			}
//		}

		public int getLevel() {
			Stack<Node> stack = new Stack<>();

			stack.push(this);

			int level = 0;
			// TODO: Using a stack here is useless...
			while (!stack.isEmpty()) {
				Node curNode = stack.pop();
				if (curNode.parent != null) {
					level++;
					stack.push(curNode.parent);
				}
			}

			return level;
		}

		public void setChild(int parentBranch, Node node) {
		}

		public int getChildIndex(Node child) {
			return -1;
		}

		public int getNumSubtrees() {
			return 1;
		}

		protected boolean skipInLevelCount() {
			return false;
		}

		public int subtreeDepth() {
			return 0;
		}
	}

	public static class FoundNode {

		public Node node;

		public SplitNode parent;

		public int parentBranch;

		public FoundNode(Node node, SplitNode parent, int parentBranch) {
			this.node = node;
			this.parent = parent;
			this.parentBranch = parentBranch;
		}
	}

	public static abstract class LearningNode extends Node {
		public LearningNode(FIMTDD tree) {
			super(tree);
		}

		public abstract void learnFromInstance(Instance inst);
	}


	public static class LeafNode extends LearningNode {

		private static final long serialVersionUID = 1L;

		// Perceptron model that carries out the actual learning in each node
		public FIMTDDPerceptron learningModel;

		protected AutoExpandVector<FIMTDDNumericAttributeClassObserver> attributeObservers = new AutoExpandVector<FIMTDDNumericAttributeClassObserver>();

		protected double examplesSeenAtLastSplitEvaluation = 0;

		/**
		 * Create a new LeafNode
		 */
		public LeafNode(FIMTDD tree) {
			super(tree);
			if (tree.buildingModelTree()) {
				learningModel = tree.newLeafModel();
			}
			examplesSeen = 0;
			sumOfValues = 0;
			sumOfSquares = 0;
			sumOfAbsErrors = 0;
		}

		public LeafNode(FIMTDD tree, Node existingNode) {
			super(tree);
			if (tree.buildingModelTree()) {
				learningModel = tree.newLeafModel();
			}
			assert !(existingNode instanceof SplitNode);
			examplesSeen = existingNode.examplesSeen;
			sumOfValues = existingNode.sumOfValues;
			sumOfSquares = existingNode.sumOfSquares;
			sumOfAbsErrors = existingNode.sumOfAbsErrors;
			parent = existingNode.parent;
		}

		public void setChild(int parentBranch, Node node) {
		}

		public int getChildIndex(Node child) {
			return -1;
		}

		public int getNumSubtrees() {
			return 1;
		}

		protected boolean skipInLevelCount() {
			return false;
		}

		/**
		 * Method to learn from an instance that passes the new instance to the perceptron learner,
		 * and also prevents the class value from being truncated to an int when it is passed to the
		 * attribute observer
		 */
		public void learnFromInstance(Instance inst) {
			assert tree.activeLeafNodeCount != 0;
			//The prediction must be calculated here -- it may be different from the tree's prediction due to alternate trees
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

			for (int i = 0; i < inst.numAttributes() - 1; i++) {
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
		}

		/**
		 * Return the best split suggestions for this node using the given split criteria
		 */
		public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion) {

			List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();

			// Set the nodeStatistics up as the preSplitDistribution, rather than the observedClassDistribution
			double[] nodeSplitDist = new double[]{examplesSeen, sumOfValues, sumOfSquares};

			if (!tree.prePruneOption.isSet()) {
				// add null split as an option
				bestSuggestions.add(new AttributeSplitSuggestion(null,
						new double[0][], criterion.getMeritOfSplit(
						nodeSplitDist,
						new double[][]{nodeSplitDist})));
			}

			for (int i = 0; i < this.attributeObservers.size(); i++) {
				FIMTDDNumericAttributeClassObserver obs = this.attributeObservers.get(i);
				if (obs != null) {

					// AT THIS STAGE NON-NUMERIC ATTRIBUTES ARE IGNORED
					AttributeSplitSuggestion bestSuggestion = null;
					if (obs instanceof FIMTDDNumericAttributeClassObserver) {
						bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion, nodeSplitDist, i, true);
					}

					if (bestSuggestion != null) {
						bestSuggestions.add(bestSuggestion);
					}
				}
			}
			return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
		}

		/**
		 * Retrieve the class votes using the perceptron learner
		 */
		public double getPredictionModel(Instance inst) {
			return learningModel.prediction(inst);
		}

		public double getPredictionTargetMean(Instance inst) {
			return (examplesSeen > 0.0) ? sumOfValues / examplesSeen : 0.0;
		}

		public double getPrediction(Instance inst) {
			return (tree.buildingModelTree()) ? getPredictionModel(inst) : getPredictionTargetMean(inst);
		}

		public void checkForSplit(FIMTDD tree) {
			// If it has seen Nmin examples since it was last tested for splitting, attempt a split of this node
			if (examplesSeen - examplesSeenAtLastSplitEvaluation >= tree.gracePeriodOption.getValue()) {
				int index = (parent != null) ? parent.getChildIndex(this) : 0;
				tree.attemptToSplit(this, parent, index);

				// Take note of how many instances were seen when this split evaluation was made, so we know when to perform the next split evaluation
				examplesSeenAtLastSplitEvaluation = examplesSeen;
			}
		}

		public void describeSubtree(StringBuilder out, int indent) {
			StringUtils.appendIndented(out, indent, "Leaf ");
			if (tree.buildingModelTree()) {
				learningModel.getModelDescription(out, 0);
			} else {
				out.append(tree.getClassNameString() + " = " + String.format("%.4f", (sumOfValues / examplesSeen)));
				StringUtils.appendNewline(out);
			}
		}
	}

	public static class InactiveLearningNode extends LearningNode {

		private static final long serialVersionUID = 1L;

		public InactiveLearningNode(LeafNode leafNode) {
			super(leafNode.tree);
			examplesSeen = leafNode.examplesSeen;
			sumOfValues = leafNode.sumOfValues;
			sumOfSquares = leafNode.sumOfSquares;
			sumOfAbsErrors = leafNode.sumOfAbsErrors;
			parent = leafNode.parent;
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
		}
	}

	public abstract static class InnerNode extends Node {
		// The InnerNode and SplitNode design is used for easy extension in ORTO
		private static final long serialVersionUID = 1L;

		protected AutoExpandVector<Node> children = new AutoExpandVector<Node>();

		// The error values for the Page Hinckley test
		// PHmT = the cumulative sum of the errors
		// PHMT = the minimum error value seen so far
		protected double PHsum = 0;
		protected double PHmin = Double.MAX_VALUE;

		// Keep track of the statistics for loss error calculations
		protected double lossExamplesSeen;
		protected double lossFadedSumOriginal;
		protected double lossFadedSumAlternate;
		protected double lossNumQiTests;
		protected double lossSumQi;
		protected double previousWeight = 0;

		public InnerNode(FIMTDD tree) {
			super(tree);
		}

		public int numChildren() {
			return children.size();
		}

		public Node getChild(int index) {
			return children.get(index);
		}

		public int getChildIndex(Node child) {
			return children.indexOf(child);
		}

		public void setChild(int index, Node child) {
			children.set(index, child);
		}

		public void disableChangeDetection() {
			changeDetection = false;
			for (Node child : children) {
				child.disableChangeDetection();
			}
		}

		public void restartChangeDetection() {
			if (alternateTree == null) {
				changeDetection = true;
				PHsum = 0;
				PHmin = Integer.MAX_VALUE;
				for (Node child : children)
					child.restartChangeDetection();
			}
		}

		@Override
		public int calcByteSizeIncludingSubtree() {
			int byteSize = calcByteSize();
			for (Node child : this.children) {
				if (child != null) {
					byteSize += child.calcByteSizeIncludingSubtree();
				}
			}
			return byteSize;
		}

		/**
		 * Check to see if the tree needs updating
		 */
		public boolean PageHinckleyTest(double error, double threshold) {
			// Update the cumulative mT sum
			PHsum += error;

			// Update the minimum mT value if the new mT is
			// smaller than the current minimum
			if (PHsum < PHmin) {
				PHmin = PHsum;
			}
			// Return true if the cumulative value - the current minimum is
			// greater than the current threshold (in which case we should adapt)
			return PHsum - PHmin > threshold;
		}

		public void initializeAlternateTree() {
			// Start a new alternate tree, beginning with a learning node
			alternateTree = tree.newLeafNode();
			alternateTree.originalNode = this;

			// Set up the blank statistics
			// Number of instances reaching this node since the alternate tree was started
			lossExamplesSeen = 0;
			// Faded squared error (original tree)
			lossFadedSumOriginal = 0;
			// Faded squared error (alternate tree)
			lossFadedSumAlternate = 0;
			// Number of evaluations of alternate tree
			lossNumQiTests = 0;
			// Sum of Qi values
			lossSumQi = 0;
			// Number of examples at last test
			previousWeight = 0;

			// Disable the change detection mechanism bellow this node
			disableChangeDetection();
		}

		@Override
		public int subtreeDepth() {
			int maxChildDepth = 0;
			for (Node child : this.children) {
				if (child != null) {
					int depth = child.subtreeDepth();
					if (depth > maxChildDepth) {
						maxChildDepth = depth;
					}
				}
			}
			return maxChildDepth + 1;
		}
	}

	public static class SplitNode extends InnerNode {

		private static final long serialVersionUID = 1L;

		protected InstanceConditionalTest splitTest;

		/**
		 * Create a new SplitNode
		 *
		 * @param tree
		 */
		public SplitNode(InstanceConditionalTest splitTest, FIMTDD tree) {
			super(tree);
			this.splitTest = splitTest;
		}

		public int instanceChildIndex(Instance inst) {
			return splitTest.branchForInstance(inst);
		}

		public Node descendOneStep(Instance inst) {
			return children.get(splitTest.branchForInstance(inst));
		}

		public void describeSubtree(StringBuilder out, int indent) {
			for (int branch = 0; branch < children.size(); branch++) {
				Node child = getChild(branch);
				if (child != null) {
					StringUtils.appendIndented(out, indent, "if ");
					out.append(this.splitTest.describeConditionForBranch(branch,
							tree.getModelContext()));
					out.append(": ");
					StringUtils.appendNewline(out);
					child.describeSubtree(out, indent + 2);
				}
			}
		}

		public double getPrediction(Instance inst) {
			return children.get(splitTest.branchForInstance(inst)).getPrediction(inst);
		}
	}

	public class FIMTDDPerceptron implements Serializable {

		private static final long serialVersionUID = 1L;

		protected FIMTDD tree;

		// The Perception weights 
		protected DoubleVector weightAttribute = new DoubleVector();

		protected double sumOfValues;
		protected double sumOfSquares;

		// The number of instances contributing to this model
		protected double instancesSeen = 0;

		// If the model should be reset or not
		protected boolean reset;

		public String getPurposeString() {
			return "A perceptron regressor as specified by Ikonomovska et al. used for FIMTDD";
		}

		public FIMTDDPerceptron(FIMTDDPerceptron original) {
			this.tree = original.tree;
			weightAttribute = (DoubleVector) original.weightAttribute.copy();
			reset = false;
		}

		public FIMTDDPerceptron(FIMTDD tree) {
			this.tree = tree;
			reset = true;
		}


		public DoubleVector getWeights() {
			return weightAttribute;
		}

		/**
		 * Update the model using the provided instance
		 */
		public void updatePerceptron(Instance inst) {

			// Initialize perceptron if necessary   
			if (reset == true) {
				reset = false;
				weightAttribute = new DoubleVector();
				instancesSeen = 0;
				for (int j = 0; j < inst.numAttributes(); j++) { // The last index corresponds to the constant b
					weightAttribute.setValue(j, 2 * tree.classifierRandom.nextDouble() - 1);
				}
			}

			// Update attribute statistics
			instancesSeen += inst.weight();

			// Update weights
			double learningRatio = 0.0;
			if (tree.learningRatioConstOption.isSet()) {
				learningRatio = learningRatioOption.getValue();
			} else {
				learningRatio = learningRatioOption.getValue() / (1 + instancesSeen * tree.learningRateDecayFactorOption.getValue());
			}

			sumOfValues += inst.weight() * inst.classValue();
			sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();

			// Loop for compatibility with bagging methods 
			for (int i = 0; i < (int) inst.weight(); i++) {
				updateWeights(inst, learningRatio);
			}
		}

		public void updateWeights(Instance inst, double learningRatio) {
			// Compute the normalized instance and the delta
			DoubleVector normalizedInstance = normalizedInstance(inst);
			double normalizedPrediction = prediction(normalizedInstance);
			double normalizedValue = tree.normalizeTargetValue(inst.classValue());
			double delta = normalizedValue - normalizedPrediction;
			normalizedInstance.scaleValues(delta * learningRatio);

			weightAttribute.addValues(normalizedInstance);
		}

		public DoubleVector normalizedInstance(Instance inst) {
			// Normalize Instance
			DoubleVector normalizedInstance = new DoubleVector();
			for (int j = 0; j < inst.numAttributes() - 1; j++) {
				int instAttIndex = modelAttIndexToInstanceAttIndex(j, inst);
				double mean = tree.sumOfAttrValues.getValue(j) / tree.weightSeen;
				double sd = computeSD(tree.sumOfAttrSquares.getValue(j), tree.sumOfAttrValues.getValue(j), tree.weightSeen);
				if (inst.attribute(instAttIndex).isNumeric() && tree.weightSeen > 1 && sd > 0)
					normalizedInstance.setValue(j, (inst.value(instAttIndex) - mean) / (3 * sd));
				else
					normalizedInstance.setValue(j, 0);
			}
			if (tree.weightSeen > 1)
				normalizedInstance.setValue(inst.numAttributes() - 1, 1.0); // Value to be multiplied with the constant factor
			else
				normalizedInstance.setValue(inst.numAttributes() - 1, 0.0);
			return normalizedInstance;
		}

		/**
		 * Output the prediction made by this perceptron on the given instance
		 */
		public double prediction(DoubleVector instanceValues) {
			return scalarProduct(weightAttribute, instanceValues);
		}

		protected double prediction(Instance inst) {
			DoubleVector normalizedInstance = normalizedInstance(inst);
			double normalizedPrediction = prediction(normalizedInstance);
			return denormalizePrediction(normalizedPrediction, tree);
		}

		private double denormalizePrediction(double normalizedPrediction, FIMTDD tree) {
			double mean = tree.sumOfValues / tree.weightSeen;
			double sd = computeSD(tree.sumOfSquares, tree.sumOfValues, tree.weightSeen);
			if (weightSeen > 1)
				return normalizedPrediction * sd * 3 + mean;
			else
				return 0.0;
		}

		public void getModelDescription(StringBuilder out, int indent) {
			StringUtils.appendIndented(out, indent, getClassNameString() + " =");
			if (getModelContext() != null) {
				for (int j = 0; j < getModelContext().numAttributes() - 1; j++) {
					if (getModelContext().attribute(j).isNumeric()) {
						out.append((j == 0 || weightAttribute.getValue(j) < 0) ? " " : " + ");
						out.append(String.format("%.4f", weightAttribute.getValue(j)));
						out.append(" * ");
						out.append(getAttributeNameString(j));
					}
				}
				out.append(" + " + weightAttribute.getValue((getModelContext().numAttributes() - 1)));
			}
			StringUtils.appendNewline(out);
		}
	}


	//endregion ================ CLASSES ================

	//region ================ METHODS ================

	// region --- Regressor methods

	public String getPurposeString() {
		return "Implementation of the FIMT-DD tree as described by Ikonomovska et al.";
	}

	public void resetLearningImpl() {
		this.treeRoot = null;
		this.leafNodeCount = 0;
		this.splitNodeCount = 0;
		this.maxID = 0;
		this.weightSeen = 0;
		this.sumOfValues = 0.0;
		this.sumOfSquares = 0.0;

		this.sumOfAttrValues = new DoubleVector();
		this.sumOfAttrSquares = new DoubleVector();
	}

	public boolean isRandomizable() {
		return true;
	}

	public void getModelDescription(StringBuilder out, int indent) {
		if (treeRoot != null) treeRoot.describeSubtree(out, indent);
	}

	protected Measurement[] getModelMeasurementsImpl() {
		double leafSizeEstimate = (
				activeLeafNodeCount * activeLeafByteSizeEstimate + inactiveLeafNodeCount * inactiveLeafByteSizeEstimate)
				* byteSizeEstimateOverheadFraction;
		// Don't measure memory size if we're not using memory management.
		// TODO: Allow this measurement to happen, we have a guard in the memory estimate function now
		int totalTreeSize = disableMemoryManagement.isSet()? 0 : this.measureByteSize();
		return new Measurement[]{
				new Measurement("tree size (nodes)", this.splitNodeCount
						+ this.activeLeafNodeCount + this.inactiveLeafNodeCount),
				new Measurement("tree size (leaves)", this.activeLeafNodeCount
						+ this.inactiveLeafNodeCount),
				new Measurement("active learning leaves",
						this.activeLeafNodeCount),
				new Measurement("inactive learning leaves",
						this.inactiveLeafNodeCount),
				new Measurement("tree depth", measureTreeDepth()),
				new Measurement("active leaf byte size estimate",
						this.activeLeafByteSizeEstimate),
				new Measurement("inactive leaf byte size estimate",
						this.inactiveLeafByteSizeEstimate),
				new Measurement("byte size estimate overhead",
						this.byteSizeEstimateOverheadFraction),
				new Measurement("total leaf byte size estimate",
						leafSizeEstimate),
				new Measurement("total tree size",
						totalTreeSize)
		};
	}

//	public int calcByteSize() {
//		return (int) SizeOf.fullSizeOf(this);
//	}

	public double[] getVotesForInstance(Instance inst) {
		if (treeRoot == null) {
			return new double[]{0};
		}

		double prediction = treeRoot.getPrediction(inst);
		if (Double.isNaN(prediction)) {
			return new double[]{0};
		}
		return new double[]{prediction};
	}

	public double normalizeTargetValue(double value) {
		if (weightSeen > 1) {
			double sd = Math.sqrt((sumOfSquares - ((sumOfValues * sumOfValues) / weightSeen)) / weightSeen);
			double average = sumOfValues / weightSeen;
			if (sd > 0 && weightSeen > 1)
				return (value - average) / (3 * sd);
			else
				return 0.0;
		}
		return 0.0;
	}

	public double getNormalizedError(Instance inst, double prediction) {
		double normalPrediction = normalizeTargetValue(prediction);
		double normalValue = normalizeTargetValue(inst.classValue());
		return Math.abs(normalValue - normalPrediction);
	}


	/**
	 * Method for updating (training) the model using a new instance
	 */
	public void trainOnInstanceImpl(Instance inst) {
		checkRoot();

		instancesSeen++;
		weightSeen += inst.weight();
		sumOfValues += inst.weight() * inst.classValue();
		sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();

		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			int aIndex = modelAttIndexToInstanceAttIndex(i, inst);
			sumOfAttrValues.addToValue(i, inst.weight() * inst.value(aIndex));
			sumOfAttrSquares.addToValue(i, inst.weight() * inst.value(aIndex) * inst.value(aIndex));
		}

		double prediction = treeRoot.getPrediction(inst);
		processInstance(inst, treeRoot, prediction, getNormalizedError(inst, prediction), false);

		if (this.instancesSeen % this.memoryEstimatePeriodOption.getValue() == 0 && !disableMemoryManagement.isSet()) {
			estimateModelByteSizes();
		}

//		FoundNode[] currentLearningNodes = findLearningNodes();
//		if (currentLearningNodes.length != activeLeafNodeCount + inactiveLeafNodeCount) {
//			System.out.println("WTF!");
//		}
	}

	public void processInstance(Instance inst, Node node, double prediction, double normalError, boolean inAlternate) {
		Node currentNode = node;
		while (true) {
			if (currentNode instanceof LearningNode) {
				((LearningNode) currentNode).learnFromInstance(inst);
				break;
			} else {
				currentNode.examplesSeen += inst.weight();
				currentNode.sumOfAbsErrors += inst.weight() * normalError;
//				SplitNode iNode = (SplitNode) currentNode; // TODO: This cast might be an issue
//				if (!inAlternate && iNode.alternateTree != null) {
//					boolean altTree = true;
//					double lossO = Math.pow(inst.classValue() - prediction, 2);
//					double lossA = Math.pow(inst.classValue() - iNode.alternateTree.getPrediction(inst), 2);
//
//					// Loop for compatibility with bagging methods
//					for (int i = 0; i < inst.weight(); i++) {
//						iNode.lossFadedSumOriginal = lossO + alternateTreeFadingFactorOption.getValue() * iNode.lossFadedSumOriginal;
//						iNode.lossFadedSumAlternate = lossA + alternateTreeFadingFactorOption.getValue() * iNode.lossFadedSumAlternate;
//						iNode.lossExamplesSeen++;
//
//						double Qi = Math.log((iNode.lossFadedSumOriginal) / (iNode.lossFadedSumAlternate));
//						iNode.lossSumQi += Qi;
//						iNode.lossNumQiTests += 1;
//					}
//					double Qi = Math.log((iNode.lossFadedSumOriginal) / (iNode.lossFadedSumAlternate));
//					double previousQiAverage = iNode.lossSumQi / iNode.lossNumQiTests;
//					double QiAverage = iNode.lossSumQi / iNode.lossNumQiTests;
//
//					if (iNode.lossExamplesSeen - iNode.previousWeight >= alternateTreeTMinOption.getValue()) {
//						iNode.previousWeight = iNode.lossExamplesSeen;
//						if (Qi > 0) {
//							// Switch the subtrees
//							Node parent = currentNode.getParent();
//
//							if (parent != null) {
//								Node replacementTree = iNode.alternateTree;
//								parent.setChild(parent.getChildIndex(currentNode), replacementTree);
//								if (growthAllowed) replacementTree.restartChangeDetection();
//							} else {
//								treeRoot = iNode.alternateTree;
//								treeRoot.restartChangeDetection();
//							}
//
//							currentNode = iNode.alternateTree;
//							currentNode.originalNode = null;
//							altTree = false;
//						} else if (
//								(QiAverage < previousQiAverage && iNode.lossExamplesSeen >= (10 * this.gracePeriodOption.getValue()))
//								|| iNode.lossExamplesSeen >= alternateTreeTimeOption.getValue()
//								) {
//							// Remove the alternate tree
//							iNode.alternateTree = null;
//							if (growthAllowed) iNode.restartChangeDetection();
//							altTree = false;
//						}
//					}
//
//					if (altTree) {
//						growthAllowed = false;
//						processInstance(inst, iNode.alternateTree, prediction, normalError,true);
//					}
//				}
//
//				if (iNode.changeDetection && !inAlternate) {
//					if (iNode.PageHinckleyTest(normalError - iNode.sumOfAbsErrors / iNode.examplesSeen - PageHinckleyAlphaOption.getValue(), PageHinckleyThresholdOption.getValue())) {
//						iNode.initializeAlternateTree();
//					}
//				}
				if (currentNode instanceof SplitNode) {
					currentNode = ((SplitNode) currentNode).descendOneStep(inst);
				}
			}
		}
	}

	// endregion --- Regressor methods

	// region --- Object instatiation methods

	public int calcByteSize() {
		int size = (int) SizeOf.sizeOf(this);
		if (this.treeRoot != null) {
			size += this.treeRoot.calcByteSizeIncludingSubtree();
		}
		return size;
	}

	@Override
	public int measureByteSize() {
		return calcByteSize();
	}

	public int measureTreeDepth() {
		if (this.treeRoot != null) {
			return this.treeRoot.subtreeDepth();
		}
		return 0;
	}

	protected FIMTDDNumericAttributeClassObserver newNumericClassObserver() {
		return new FIMTDDNumericAttributeClassObserver();
	}

	protected SplitNode newSplitNode(InstanceConditionalTest splitTest) {
		maxID++;
		return new SplitNode(splitTest, this);
	}

	protected LeafNode newLeafNode() {
		maxID++;
		return new LeafNode(this);
	}

	protected LeafNode newLeafNode(Node existingNode) {
		maxID++;
		return new LeafNode(this, existingNode);
	}

	protected InactiveLearningNode createInactiveNode(LeafNode existingNode) {
		return new InactiveLearningNode(existingNode);
	}

	protected FIMTDDPerceptron newLeafModel() {
		return new FIMTDDPerceptron(this);
	}

	//endregion --- Object instatiation methods

	// region --- Processing methods


	protected void checkRoot() {
		if (treeRoot == null) {
			treeRoot = newLeafNode();
			leafNodeCount = 1;
			activeLeafNodeCount = 1;
		}
	}

	public static double computeHoeffdingBound(double range, double confidence, double n) {
		return Math.sqrt(((range * range) * Math.log(1 / confidence)) / (2.0 * n));
	}

	public boolean buildingModelTree() {
		return !regressionTreeOption.isSet();
	}

	protected void attemptToSplit(LeafNode node, Node parent, int parentIndex) {
		// Set the split criterion to use to the SDR split criterion as described by Ikonomovska et al. 
		SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);

		// Using this criterion, find the best split per attribute and rank the results
		AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion);
		Arrays.sort(bestSplitSuggestions);

		// Declare a variable to determine if any of the splits should be performed
		boolean shouldSplit = false;

		// If only one split was returned, use it
		if (bestSplitSuggestions.length < 2) {
			shouldSplit = bestSplitSuggestions.length > 0;
		} else { // Otherwise, consider which of the splits proposed may be worth trying

			// Determine the Hoeffding bound value, used to select how many instances should be used to make a test decision
			// to feel reasonably confident that the test chosen by this sample is the same as what would be chosen using infinite examples
			double hoeffdingBound = computeHoeffdingBound(1, this.splitConfidenceOption.getValue(), node.examplesSeen);
			// Determine the top two ranked splitting suggestions
			AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
			AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];

			// If the upper bound of the sample mean for the ratio of SDR(best suggestion) to SDR(second best suggestion),
			// as determined using the Hoeffding bound, is less than 1, then the true mean is also less than 1, and thus at this
			// particular moment of observation the bestSuggestion is indeed the best split option with confidence 1-delta, and
			// splitting should occur.
			// Alternatively, if two or more splits are very similar or identical in terms of their splits, then a threshold limit
			// (default 0.05) is applied to the Hoeffding bound; if the Hoeffding bound is smaller than this limit then the two
			// competing attributes are equally good, and the split will be made on the one with the higher SDR value.
			if ((secondBestSuggestion.merit / bestSuggestion.merit < 1 - hoeffdingBound) || (hoeffdingBound < this.tieThresholdOption.getValue())) {
				shouldSplit = true;
			}
			// If the splitting criterion was not met, initiate pruning of the E-BST structures in each attribute observer
			else {
				for (int i = 0; i < node.attributeObservers.size(); i++) {
					FIMTDDNumericAttributeClassObserver obs = node.attributeObservers.get(i);
					if (obs != null) {
						obs.removeBadSplits(splitCriterion, secondBestSuggestion.merit / bestSuggestion.merit, bestSuggestion.merit, hoeffdingBound);
					}
				}
			}
		}

		// If the splitting criterion were met, split the current node using the chosen attribute test, and
		// make two new branches leading to (empty) leaves
		if (shouldSplit) {
			AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];

			if (splitDecision.splitTest == null) {
				//preprune - null wins
				assert false;
				deactivateLearningNode(node, (SplitNode) parent, parentIndex);
			} else {
				SplitNode newSplit = newSplitNode(splitDecision.splitTest);
				newSplit.copyStatistics(node);
				newSplit.changeDetection = node.changeDetection;
				newSplit.ID = node.ID;
				leafNodeCount--;
				for (int i = 0; i < splitDecision.numSplits(); i++) {
					LeafNode newChild = newLeafNode();
					if (buildingModelTree()) {
						// Copy the splitting node's perceptron to it's children
						newChild.learningModel = new FIMTDDPerceptron(node.learningModel);

					}
					newChild.changeDetection = node.changeDetection;
					newChild.setParent(newSplit);
					newSplit.setChild(i, newChild);
					leafNodeCount++;
				}
				if (parent == null && node.originalNode == null) {
					treeRoot = newSplit;
				} else if (parent == null) {
					node.originalNode.alternateTree = newSplit;
				} else {
					parent.setChild(parentIndex, newSplit);
					newSplit.setParent(parent);
				}

				activeLeafNodeCount--;
				decisionNodeCount++;
				activeLeafNodeCount += splitDecision.numSplits();
				splitNodeCount++;
			}
			// manage memory
			if (!disableMemoryManagement.isSet()) {
				enforceTrackerLimit(Double.NaN);
			}

		}
	}

	/**
	 * Enforce memory management for tree.
	 * See Data Stream Mining: A practical approach, section 3.3 Memory Management for details
	 */
	public void enforceTrackerLimit(Double actualModelSize) {
		double leafSizeEstimate = (
				activeLeafNodeCount * activeLeafByteSizeEstimate + inactiveLeafNodeCount * inactiveLeafByteSizeEstimate)
				* byteSizeEstimateOverheadFraction;
		if (!actualModelSize.isNaN()) {
			leafSizeEstimate = actualModelSize;
		}

		int leafSumBefore = activeLeafNodeCount + inactiveLeafNodeCount;
//		if (leafSizeEstimate > maxByteSizeOption.getValue()) {
//			System.out.println(treeID + ": Max byte size limit hit at: " + leafSizeEstimate + " bytes!");
//			System.out.println(treeID + ": Limit was: " + maxByteSizeOption.getValue() + " bytes");
//			System.out.println(treeID + ": Active leaf nodes before fixing: " + activeLeafNodeCount);
//			System.out.println(treeID + ": Inactive leaf nodes before fixing: " + inactiveLeafNodeCount);
//		}

		boolean nodesDeactivated = false;

		if (stopMemManagementOption.isSet()) {
			if (leafSizeEstimate > maxByteSizeOption.getValue() && activeLeafNodeCount != 0) {
//				System.out.println(treeID + ": Stopping growth...");
				//System.out.println(treeID + ": Hit growth limit, stopping growth of tree");
				growthAllowed = false;
				FoundNode[] learningNodes = findLearningNodes();
				// Deactivate all learning nodes
				nodesDeactivated = true;
				for (FoundNode learningNode : learningNodes) {
					if (learningNode.node instanceof LeafNode) {
						deactivateLearningNode((LeafNode) learningNode.node, learningNode.parent, learningNode.parentBranch);
					}
				}
//				System.out.println(treeID + ": Active leaf nodes after deactivation: " + activeLeafNodeCount);
//				System.out.println(treeID + ": Inactive leaf nodes after deactivation: " + inactiveLeafNodeCount);
				leafNumAtDeactivate = activeLeafNodeCount + inactiveLeafNodeCount;
			} else if (!growthAllowed && maxByteSizeOption.getValue() > (leafSizeEstimate  + activeLeafByteSizeEstimate)) {
				assert leafNumAtDeactivate == activeLeafNodeCount + inactiveLeafNodeCount;
//				System.out.println(treeID + ": Enabling growth again!");
				// Reactivate nodes
				FoundNode[] learningNodes = findLearningNodes();
				growthAllowed = true;
				for (FoundNode learningNode : learningNodes) {
					if (learningNode.node instanceof InactiveLearningNode) {
						activateLearningNode((InactiveLearningNode) learningNode.node, learningNode.parent, learningNode.parentBranch);
						FoundNode[] currentLearningNodes = findLearningNodes();
						assert currentLearningNodes.length == leafSumBefore: "Num leafs before: " + leafSumBefore + ". Num leafs now: " + currentLearningNodes.length;
					}
				}
//				System.out.println(treeID + ": Active leaf nodes after re-activation: " + activeLeafNodeCount);
//				System.out.println(treeID + ": Inactive leaf nodes after re-activation: " + inactiveLeafNodeCount);
			}

			return;
		}
//		assert false : "Should not get here!";
		if ((inactiveLeafNodeCount > 0) || (leafSizeEstimate > maxByteSizeOption.getValue())) {
			nodesDeactivated = true;
			// Get all learning nodes and sort them by promise
			FoundNode[] learningNodes = findLearningNodes();
			Arrays.sort(learningNodes, Comparator.comparingDouble(foundNode -> foundNode.node.calculatePromise()));
			int maxActive = 0;
			// Find the maximum number of active nodes under current memory budget
			while (maxActive < learningNodes.length) {
				maxActive++;
				double curMaxEstimate = (maxActive * activeLeafByteSizeEstimate + (learningNodes.length - maxActive)
						* inactiveLeafByteSizeEstimate);
				double curEstimateWithOverhead = curMaxEstimate * byteSizeEstimateOverheadFraction;
				if (curEstimateWithOverhead > maxByteSizeOption.getValue()) {
					maxActive--;
					break;
				}
			}
			if (maxActive < 1) {
				maxActive = 1;
			}
			// Deactivate least promising active nodes
			int cutoff = learningNodes.length - maxActive;
			for (int i = 0; i < cutoff; i++) {
				if (!(learningNodes[i].node instanceof InactiveLearningNode)) {
					assert learningNodes[i].node instanceof LeafNode;
					deactivateLearningNode(
							(LeafNode) learningNodes[i].node,
							learningNodes[i].parent,
							learningNodes[i].parentBranch);
				}
			}
			// Reactivate most promising inactive modes
			for (int i = cutoff; i < learningNodes.length; i++) {
				if (learningNodes[i].node instanceof InactiveLearningNode) {
					activateLearningNode(
							(InactiveLearningNode) learningNodes[i].node,
							learningNodes[i].parent,
							learningNodes[i].parentBranch);
				}
			}
		}
//		if (nodesDeactivated) {
//			System.out.println(treeID + ": Active leaf nodes after fixing: " + activeLeafNodeCount);
//			System.out.println(treeID + ": Inactive leaf nodes after fixing: " + inactiveLeafNodeCount);
//		}

		int leafSumAfter = activeLeafNodeCount + inactiveLeafNodeCount;

		assert leafSumAfter == leafSumBefore : "Leaf sums were different before and after deactivation!";
	}

	protected void deactivateLearningNode(LeafNode toDeactivate, SplitNode parent, int parentBranch) {
		Node inactiveNode = createInactiveNode(toDeactivate);
		if (parent == null) {
			this.treeRoot = inactiveNode;
		} else {
			parent.setChild(parentBranch, inactiveNode);
		}
		this.activeLeafNodeCount--;
		this.inactiveLeafNodeCount++;
	}

	protected void activateLearningNode(InactiveLearningNode toActivate,
																			SplitNode parent, int parentBranch) {
		Node newLeaf = newLeafNode(toActivate);
		if (parent == null) {
			this.treeRoot = newLeaf;
		} else {
			parent.setChild(parentBranch, newLeaf);
//			System.out.println("New level: " + newLeaf.getLevel());
//			System.out.println("Existing level: " + toActivate.getLevel());
		}
		this.activeLeafNodeCount++;
		this.inactiveLeafNodeCount--;
	}

//	protected FoundNode[] findLearningNodes() {
//		List<FoundNode> foundList = new LinkedList<>();
//		findLearningNodes(this.treeRoot, null, -1, foundList);
//		return foundList.toArray(new FoundNode[foundList.size()]);
//	}


	protected FoundNode[] findLearningNodes() {

		Stack<FoundNode> nodeStack = new Stack<>();
		List<FoundNode> foundList = new ArrayList<>(inactiveLeafNodeCount + activeLeafNodeCount);

		nodeStack.push(new FoundNode(this.treeRoot, null, -1));
		while (!nodeStack.isEmpty()) {
			FoundNode curFoundNode = nodeStack.pop();
			Node curNode = curFoundNode.node;
			if (curNode instanceof SplitNode) {
				SplitNode splitNode = (SplitNode) curNode;
				for (int i = 0; i < splitNode.numChildren(); i++) {
					nodeStack.push(new FoundNode(splitNode.getChild(i), splitNode, i));
				}
			} else {
				foundList.add(curFoundNode);
			}
		}

		return foundList.toArray(new FoundNode[foundList.size()]);
	}

	/**
	 * Recurse down the tree to find leafs, either active or inactive
	 *
	 * @param node         The starting node
	 * @param parent       The starting node's parent
	 * @param parentBranch The ID of the parent branch
	 * @param found        A list containing all the leafs (learning nodes) of the tree
	 */
	protected void findLearningNodes(Node node, SplitNode parent,
																	 int parentBranch, List<FoundNode> found) {
		if (node != null) {
			if (node instanceof LearningNode) {
				found.add(new FoundNode(node, parent, parentBranch));
			}
			if (node instanceof SplitNode) {
				SplitNode splitNode = (SplitNode) node;
				for (int i = 0; i < splitNode.numChildren(); i++) {
					findLearningNodes(splitNode.getChild(i), splitNode, i,
							found);
				}
			}
		}
	}

	public void estimateModelByteSizes() {
		FoundNode[] learningNodes = findLearningNodes();
		long totalActiveSize = 0;
		long totalInactiveSize = 0;
		for (FoundNode foundNode : learningNodes) {
			if (foundNode.node instanceof InactiveLearningNode) {
				totalInactiveSize += SizeOf.fullSizeOf(foundNode.node);
			} else {
				totalActiveSize += SizeOf.fullSizeOf(foundNode.node);
			}
		}
		if (totalActiveSize > 0) {
			this.activeLeafByteSizeEstimate = (double) totalActiveSize
					/ this.activeLeafNodeCount;
		}
		if (totalInactiveSize > 0) {
			this.inactiveLeafByteSizeEstimate = (double) totalInactiveSize
					/ this.inactiveLeafNodeCount;
		}
		int actualModelSize = this.measureByteSize();
//		System.out.println(treeID + ": Actual model size: " + actualModelSize);
		double estimatedModelSize = (this.activeLeafNodeCount
				* this.activeLeafByteSizeEstimate + this.inactiveLeafNodeCount
				* this.inactiveLeafByteSizeEstimate);
//		System.out.println(treeID + ": Estimated model size: " + estimatedModelSize);
		this.byteSizeEstimateOverheadFraction = actualModelSize
				/ estimatedModelSize;
		if (actualModelSize > this.maxByteSizeOption.getValue() || !growthAllowed) {
			enforceTrackerLimit((double) actualModelSize);
		}
	}

	public double computeSD(double squaredVal, double val, double size) {
		if (size > 1)
			return Math.sqrt((squaredVal - ((val * val) / size)) / size);
		else
			return 0.0;
	}

	public double scalarProduct(DoubleVector u, DoubleVector v) {
		double ret = 0.0;
		for (int i = 0; i < Math.max(u.numValues(), v.numValues()); i++) {
			ret += u.getValue(i) * v.getValue(i);
		}
		return ret;
	}
	//endregion --- Processing methods

	//endregion ================ METHODS ================
}

