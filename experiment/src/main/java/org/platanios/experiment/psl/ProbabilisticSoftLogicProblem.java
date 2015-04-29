package org.platanios.experiment.psl;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.commons.lang3.ArrayUtils;
import org.platanios.learn.Utilities;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;
import org.platanios.learn.optimization.NewtonSolver;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.MaxFunction;
import org.platanios.learn.optimization.function.SumFunction;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class ProbabilisticSoftLogicProblem {

    private ProbabilisticSoftLogicProblem(
            BiMap<Integer, Integer> externalId2InternalId,
            ProbabilisticSoftLogicFunction function) {
        this.externalId2InternalId = HashBiMap.create(externalId2InternalId);
        this.function = function;
    }

    public static class Builder {

        public Builder(int[] observedVariableIds,
                       double[] observedVariableValues,
                       int expectedSize) {

            if ((observedVariableIds == null) != (observedVariableValues == null)) {
                throw new IllegalArgumentException("observedVariableIds does not match observedVariableValues");
            }

            this.observedVariableValues = new HashMap<>();
            if (observedVariableIds != null) {

                if (observedVariableIds.length != observedVariableValues.length) {
                    throw new IllegalArgumentException("observedVariableIds does not match observedVariableValues");
                }

                for (int i = 0; i < observedVariableIds.length; ++i) {
                    this.observedVariableValues.put(observedVariableIds[i], observedVariableValues[i]);
                }
            }

            this.externalId2InternalId = HashBiMap.create(expectedSize);
            this.functions = new ArrayList<>();

        }

        public ProbabilisticSoftLogicProblem.Builder addRule(int[] headVariableIds,
                                                              int[] bodyVariableIds,
                                                              boolean[] headNegations,
                                                              boolean[] bodyNegations,
                                                              double power,
                                                              double weight) {

            RulePart headPart = this.convertToInternalRepresentation(headVariableIds, headNegations, true);
            RulePart bodyPart = this.convertToInternalRepresentation(bodyVariableIds, bodyNegations, false);
            double ruleMaximumValue = 1 + headPart.ObservedConstant + bodyPart.ObservedConstant;

            if (ruleMaximumValue == 0) {
                return this;
            }

            int[] variablesIndexes = Utilities.union(headPart.InternalVariableIds, bodyPart.InternalVariableIds);
            LinearFunction linearFunction = new LinearFunction(Vectors.dense(variablesIndexes.length), ruleMaximumValue);

            for (int headVariable = 0; headVariable < headPart.InternalVariableIds.length; headVariable++) {
                Vector coefficients = Vectors.dense(variablesIndexes.length);
                if (headPart.Negations[headVariable]) {
                    coefficients.set(ArrayUtils.indexOf(variablesIndexes, headPart.InternalVariableIds[headVariable]), 1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, -1));
                } else {
                    coefficients.set(ArrayUtils.indexOf(variablesIndexes, headPart.InternalVariableIds[headVariable]), -1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, 0));
                }
            }
            for (int bodyVariable = 0; bodyVariable < bodyPart.InternalVariableIds.length; bodyVariable++) {
                Vector coefficients = Vectors.dense(variablesIndexes.length);
                if (bodyPart.Negations[bodyVariable]) {
                    coefficients.set(ArrayUtils.indexOf(variablesIndexes, bodyPart.InternalVariableIds[bodyVariable]), -1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, 0));
                } else {
                    coefficients.set(ArrayUtils.indexOf(variablesIndexes, bodyPart.InternalVariableIds[bodyVariable]), 1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, -1));
                }
            }

            this.functions.add(new FunctionContainer(linearFunction, variablesIndexes, weight, power));

            return this;
        }

        public ProbabilisticSoftLogicProblem build() {

            SumFunction.Builder sumFunctionBuilder = new SumFunction.Builder(this.externalId2InternalId.size());

            for (FunctionContainer function : this.functions) {

                MaxFunction.Builder maxFunctionBuilder = new MaxFunction.Builder(this.externalId2InternalId.size());
                maxFunctionBuilder.addConstantTerm(0);
                maxFunctionBuilder.addFunctionTerm(function.LinearFunction);
                sumFunctionBuilder.addTerm(
                        new ProbabilisticSoftLogicSumFunctionTerm(
                                maxFunctionBuilder.build(),
                                function.Power,
                                function.Weight),
                        function.VariableIndices
                );

            }

            return new ProbabilisticSoftLogicProblem(
                    this.externalId2InternalId,
                    new ProbabilisticSoftLogicFunction(sumFunctionBuilder));

        }

        private static class FunctionContainer {

            public FunctionContainer(LinearFunction linearFunction, int[] variableIndices, double weight, double power) {
                this.LinearFunction = linearFunction;
                this.VariableIndices = variableIndices;
                this.Weight = weight;
                this.Power = power;
            }

            public final double Power;
            public final double Weight;
            public final LinearFunction LinearFunction;
            public final int[] VariableIndices;

        }

        private static class RulePart {

            public RulePart(int[] internalVariableIds, boolean[] negations, double observedConstant) {
                this.InternalVariableIds = internalVariableIds;
                this.Negations = negations;
                this.ObservedConstant = observedConstant;
            }

            public final int[] InternalVariableIds;
            public final boolean[] Negations;
            public final double ObservedConstant;

        }

        private RulePart convertToInternalRepresentation(int[] externalIds, boolean[] negations, boolean isHead) {

            ArrayList<Integer> internalVariableIndexes = new ArrayList<>();
            ArrayList<Boolean> internalVariableNegations = new ArrayList<>();
            double observedConstant = 0;

            for (int i = 0; i < externalIds.length; ++i) {

                double observedValue = this.observedVariableValues.getOrDefault(externalIds[i], Double.NaN);
                if (!Double.isNaN(observedValue)) {

                    if (isHead == negations[i]) { // head negation or body positive
                        observedConstant += observedValue - 1;
                    } else {
                        observedConstant -= observedValue;
                    }

                } else {
                    int internalId = this.externalId2InternalId.getOrDefault(externalIds[i], -1);
                    if (internalId < 0) {
                        internalId = this.nextInternalId++;
                        this.externalId2InternalId.put(externalIds[i], internalId);
                    }
                    internalVariableIndexes.add(internalId);
                    internalVariableNegations.add(negations[i]);
                }

            }

            int[] internalIds = new int[internalVariableIndexes.size()];
            boolean[] internalNegations = new boolean[internalVariableIndexes.size()];
            for (int i = 0; i < internalVariableIndexes.size(); ++i) {
                internalIds[i] = internalVariableIndexes.get(i);
                internalNegations[i] = internalVariableNegations.get(i);
            }
            return new RulePart(internalIds, internalNegations, observedConstant);
        }

        private final HashMap<Integer, Double> observedVariableValues;
        private final BiMap<Integer, Integer> externalId2InternalId;
        private final ArrayList<FunctionContainer> functions;
        private int nextInternalId = 0;
    }

    private static final class ProbabilisticSoftLogicFunction extends SumFunction {

        private ProbabilisticSoftLogicFunction(SumFunction.Builder sumFunctionBuilder) {
            super(sumFunctionBuilder);
        }

        public LinearFunction getTermLinearFunction(int term) {
            return ((ProbabilisticSoftLogicSumFunctionTerm) terms.get(term)).getLinearFunction();
        }

        public double getTermPower(int term) {
            return ((ProbabilisticSoftLogicSumFunctionTerm) terms.get(term)).getPower();
        }

        public double getTermWeight(int term) {
            return ((ProbabilisticSoftLogicSumFunctionTerm) terms.get(term)).getWeight();
        }

    }

    private static class ProbabilisticSoftLogicSubProblemObjectiveFunction extends AbstractFunction {
        private final LinearFunction linearFunction;
        private final double power;
        private final double weight;

        private ProbabilisticSoftLogicSubProblemObjectiveFunction(LinearFunction linearFunction,
                                                                  double power,
                                                                  double weight) {
            this.linearFunction = linearFunction;
            this.power = power;
            this.weight = weight;
        }

        @Override
        public double computeValue(Vector point) {
            return weight * Math.pow(linearFunction.computeValue(point), power);
        }

        @Override
        public Vector computeGradient(Vector point) {
            if (power > 0) {
                return linearFunction.computeGradient(point).mult(
                        weight * power * Math.pow(linearFunction.computeValue(point), power - 1)
                );
            } else {
                return Vectors.build(point.size(), point.type());
            }
        }

        @Override
        public Matrix computeHessian(Vector point) {
            if (power > 1) {
                Vector a = linearFunction.computeGradient(point);
                return Matrix.generateDiagonalMatrix(a.multElementwise(a).getDenseArray())
                        .multiply(weight * power * (power - 1) * Math.pow(linearFunction.computeValue(point), power - 2));
            } else {
                return new Matrix(point.size(), point.size());
            }
        }
    }

    private static class ProbabilisticSoftLogicSumFunctionTerm extends AbstractFunction {
        private final MaxFunction maxFunction;
        private final double power;
        private final double weight;

        public ProbabilisticSoftLogicSumFunctionTerm(MaxFunction maxFunction, double power, double weight) {
            this.maxFunction = maxFunction;
            this.power = power;
            this.weight = weight;
        }

        @Override
        public double computeValue(Vector point) {
            return weight * Math.pow(maxFunction.getValue(point), power);
        }

        public MaxFunction getMaxFunction() {
            return maxFunction;
        }

        public LinearFunction getLinearFunction() {
            return (LinearFunction) maxFunction.getFunctionTerm(0);
        }

        public double getPower() {
            return power;
        }

        public double getWeight() {
            return weight;
        }

        public ProbabilisticSoftLogicSubProblemObjectiveFunction getSubProblemObjectiveFunction() {
            return new ProbabilisticSoftLogicSubProblemObjectiveFunction(
                    (LinearFunction) maxFunction.getFunctionTerm(0),
                    power,
                    weight
            );
        }
    }

    public static void solveProbabilisticSoftLogicSubProblem(
            ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblem subProblem
    ) {
        ProbabilisticSoftLogicSumFunctionTerm objectiveTerm =
                (ProbabilisticSoftLogicSumFunctionTerm) subProblem.objectiveTerm;
        if (objectiveTerm.getLinearFunction().getValue(subProblem.variables) > 0) {
            subProblem.variables.set(
                    new NewtonSolver.Builder(
                            new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemObjectiveFunction(
                                    objectiveTerm.getSubProblemObjectiveFunction(),
                                    subProblem.consensusVariables,
                                    subProblem.multipliers,
                                    subProblem.augmentedLagrangianParameter
                            ),
                            subProblem.variables).build().solve()
            );
            if (objectiveTerm.getLinearFunction().getValue(subProblem.variables) < 0) {
                subProblem.variables.set(
                        objectiveTerm.getLinearFunction().projectToHyperplane(subProblem.consensusVariables)
                );
            }
        }
    }

    private final BiMap<Integer, Integer> externalId2InternalId;
    private final ProbabilisticSoftLogicFunction function;

}
