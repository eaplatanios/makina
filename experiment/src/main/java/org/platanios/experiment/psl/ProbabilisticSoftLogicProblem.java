package org.platanios.experiment.psl;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.primitives.Booleans;
import com.google.common.primitives.Ints;
import org.apache.commons.lang3.ArrayUtils;
import org.platanios.learn.Utilities;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;
import org.platanios.learn.optimization.NewtonSolver;
import org.platanios.learn.optimization.constraint.AbstractConstraint;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.MaxFunction;
import org.platanios.learn.optimization.function.SumFunction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class ProbabilisticSoftLogicProblem {
    private final BiMap<Integer, Integer> externalToInternalIndexesMapping;
    private final ProbabilisticSoftLogicFunction objectiveFunction;
    private final List<Constraint> constraints;

    public static final class Builder {
        private final BiMap<Integer, Integer> externalToInternalIndexesMapping;

        private final Map<Integer, Double> observedVariableValues = new HashMap<>();
        private final List<FunctionTerm> functionTerms = new ArrayList<>();
        private final List<Constraint> constraints = new ArrayList<>();
        private int nextInternalIndex = 0;

        public Builder(int[] observedVariableIndexes,
                       double[] observedVariableValues,
                       int numberOfUnobservedVariables) {
            if ((observedVariableIndexes == null) != (observedVariableValues == null))
                throw new IllegalArgumentException(
                        "The provided indexes for the observed variables must much the corresponding provided values."
                );
            if (observedVariableIndexes != null) {
                if (observedVariableIndexes.length != observedVariableValues.length)
                    throw new IllegalArgumentException(
                            "The provided indexes array for the observed variables must " +
                                    "have the same length the corresponding provided values array."
                    );
                for (int i = 0; i < observedVariableIndexes.length; ++i)
                    this.observedVariableValues.put(observedVariableIndexes[i], observedVariableValues[i]);
            }
            externalToInternalIndexesMapping = HashBiMap.create(numberOfUnobservedVariables);
        }

        public Builder addRule(int[] headVariableIndexes,
                               int[] bodyVariableIndexes,
                               boolean[] headNegations,
                               boolean[] bodyNegations,
                               double power,
                               double weight) {
            RulePart headPart = convertRulePartToInternalRepresentation(headVariableIndexes, headNegations, true);
            RulePart bodyPart = convertRulePartToInternalRepresentation(bodyVariableIndexes, bodyNegations, false);
            double ruleMaximumValue = 1 + headPart.observedConstant + bodyPart.observedConstant;
            if (ruleMaximumValue == 0)
                return this;
            int[] variableIndexes = Utilities.union(headPart.variableIndexes, bodyPart.variableIndexes);
            if (variableIndexes.length == 0)
                return this;
            LinearFunction linearFunction = new LinearFunction(Vectors.dense(variableIndexes.length), ruleMaximumValue);
            for (int headVariable = 0; headVariable < headPart.variableIndexes.length; headVariable++) {
                Vector coefficients = Vectors.dense(variableIndexes.length);
                if (headPart.negations[headVariable]) {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, headPart.variableIndexes[headVariable]), 1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, -1));
                } else {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, headPart.variableIndexes[headVariable]), -1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, 0));
                }
            }
            for (int bodyVariable = 0; bodyVariable < bodyPart.variableIndexes.length; bodyVariable++) {
                Vector coefficients = Vectors.dense(variableIndexes.length);
                if (bodyPart.negations[bodyVariable]) {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, bodyPart.variableIndexes[bodyVariable]), -1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, 0));
                } else {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, bodyPart.variableIndexes[bodyVariable]), 1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, -1));
                }
            }
            functionTerms.add(new FunctionTerm(variableIndexes, linearFunction, weight, power));
            return this;
        }

        public Builder addConstraint(AbstractConstraint constraint, int... externalVariableIndexes) {
            List<Integer> internalVariableIndexes = new ArrayList<>();
            for (int externalVariableIndex : externalVariableIndexes) {
                int internalVariableIndex = externalToInternalIndexesMapping.getOrDefault(externalVariableIndex, -1);
                if (internalVariableIndex < 0) {
                    internalVariableIndex = nextInternalIndex++;
                    externalToInternalIndexesMapping.put(externalVariableIndex, internalVariableIndex);
                }
                internalVariableIndexes.add(internalVariableIndex);
            }
            constraints.add(new Constraint(constraint, Ints.toArray(internalVariableIndexes)));
            return this;
        }

        public ProbabilisticSoftLogicProblem build() {
            return new ProbabilisticSoftLogicProblem(this);
        }

        private RulePart convertRulePartToInternalRepresentation(int[] externalVariableIndexes,
                                                                 boolean[] negations,
                                                                 boolean isRuleHeadVariable) {
            List<Integer> internalVariableIndexes = new ArrayList<>();
            List<Boolean> internalVariableNegations = new ArrayList<>();
            double observedConstant = 0;
            for (int i = 0; i < externalVariableIndexes.length; ++i) {
                double observedValue = observedVariableValues.getOrDefault(externalVariableIndexes[i], Double.NaN);
                if (!Double.isNaN(observedValue)) {
                    if (isRuleHeadVariable == negations[i])
                        observedConstant += observedValue - 1;
                    else
                        observedConstant -= observedValue;
                } else {
                    int internalVariableIndex =
                            externalToInternalIndexesMapping.getOrDefault(externalVariableIndexes[i], -1);
                    if (internalVariableIndex < 0) {
                        internalVariableIndex = nextInternalIndex++;
                        externalToInternalIndexesMapping.put(externalVariableIndexes[i], internalVariableIndex);
                    }
                    internalVariableIndexes.add(internalVariableIndex);
                    internalVariableNegations.add(negations[i]);
                }
            }
            return new RulePart(
                    Ints.toArray(internalVariableIndexes),
                    Booleans.toArray(internalVariableNegations),
                    observedConstant
            );
        }

        private static class RulePart {
            private final int[] variableIndexes;
            private final boolean[] negations;
            private final double observedConstant;

            private RulePart(int[] variableIndexes,
                            boolean[] negations,
                            double observedConstant) {
                this.variableIndexes = variableIndexes;
                this.negations = negations;
                this.observedConstant = observedConstant;
            }
        }

        private static class FunctionTerm {
            private final LinearFunction linearFunction;
            private final int[] variableIndexes;
            private final double power;
            private final double weight;

            private FunctionTerm(int[] variableIndexes,
                                LinearFunction linearFunction,
                                double weight,
                                double power) {
                this.linearFunction = linearFunction;
                this.variableIndexes = variableIndexes;
                this.power = power;
                this.weight = weight;
            }
        }
    }

    private ProbabilisticSoftLogicProblem(Builder builder) {
        externalToInternalIndexesMapping = HashBiMap.create(builder.externalToInternalIndexesMapping);
        SumFunction.Builder sumFunctionBuilder = new SumFunction.Builder(externalToInternalIndexesMapping.size());
        for (Builder.FunctionTerm function : builder.functionTerms) {
            MaxFunction.Builder maxFunctionBuilder = new MaxFunction.Builder(externalToInternalIndexesMapping.size());
            maxFunctionBuilder.addConstantTerm(0);
            maxFunctionBuilder.addFunctionTerm(function.linearFunction);
            sumFunctionBuilder.addTerm(
                    new ProbabilisticSoftLogicSumFunctionTerm(
                            maxFunctionBuilder.build(),
                            function.power,
                            function.weight),
                    function.variableIndexes
            );
        }
        objectiveFunction = new ProbabilisticSoftLogicFunction(sumFunctionBuilder);
        constraints = builder.constraints;
    }

    public Map<Integer, Double> solve() {
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder solverBuilder =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(
                        objectiveFunction,
                        Vectors.dense(objectiveFunction.getNumberOfVariables())
                )
                        .subProblemSolver(ProbabilisticSoftLogicProblem::solveProbabilisticSoftLogicSubProblem)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .logObjectiveValue(false)
                        .logGradientNorm(false)
                        .loggingLevel(3);
        for (Constraint constraint : constraints)
            solverBuilder.addConstraint(constraint.constraint, constraint.variableIndexes);
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver = solverBuilder.build();
        Vector solverResult = solver.solve();
        Map<Integer, Double> inferredValues = new HashMap<>(solverResult.size());
        for (int internalVariableIndex = 0; internalVariableIndex < solverResult.size(); internalVariableIndex++)
            inferredValues.put(externalToInternalIndexesMapping.inverse().get(internalVariableIndex),
                               solverResult.get(internalVariableIndex));
        return inferredValues;
    }

    private static void solveProbabilisticSoftLogicSubProblem(
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

    private static final class ProbabilisticSoftLogicSubProblemObjectiveFunction extends AbstractFunction {
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
                return Matrix.generateDiagonalMatrix(a.multElementwise(a).getDenseArray()).multiply(
                        weight * power * (power - 1) * Math.pow(linearFunction.computeValue(point), power - 2)
                );
            } else {
                return new Matrix(point.size(), point.size());
            }
        }
    }

    private static final class ProbabilisticSoftLogicSumFunctionTerm extends AbstractFunction {
        private final MaxFunction maxFunction;
        private final double power;
        private final double weight;

        private ProbabilisticSoftLogicSumFunctionTerm(MaxFunction maxFunction, double power, double weight) {
            this.maxFunction = maxFunction;
            this.power = power;
            this.weight = weight;
        }

        @Override
        public double computeValue(Vector point) {
            return weight * Math.pow(maxFunction.getValue(point), power);
        }

        private MaxFunction getMaxFunction() {
            return maxFunction;
        }

        private LinearFunction getLinearFunction() {
            return (LinearFunction) maxFunction.getFunctionTerm(0);
        }

        private double getPower() {
            return power;
        }

        private double getWeight() {
            return weight;
        }

        private ProbabilisticSoftLogicSubProblemObjectiveFunction getSubProblemObjectiveFunction() {
            return new ProbabilisticSoftLogicSubProblemObjectiveFunction(
                    (LinearFunction) maxFunction.getFunctionTerm(0),
                    power,
                    weight
            );
        }
    }

    private static final class Constraint {
        private final AbstractConstraint constraint;
        private final int[] variableIndexes;

        private Constraint(AbstractConstraint constraint, int[] variableIndexes) {
            this.constraint = constraint;
            this.variableIndexes = variableIndexes;
        }
    }
}
