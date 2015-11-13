package org.platanios.learn.logic;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableSet;
import org.apache.commons.lang3.ArrayUtils;
import org.platanios.learn.logic.formula.Disjunction;
import org.platanios.learn.logic.formula.Formula;
import org.platanios.learn.logic.formula.Negation;
import org.platanios.learn.logic.formula.Predicate;
import org.platanios.learn.logic.grounding.GroundPredicate;
import org.platanios.learn.logic.grounding.InMemoryLazyGrounding;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;
import org.platanios.learn.optimization.NewtonSolver;
import org.platanios.learn.optimization.constraint.AbstractConstraint;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.MaxFunction;
import org.platanios.learn.optimization.function.SumFunction;
import org.platanios.learn.optimization.linesearch.NoLineSearch;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class ProbabilisticSoftLogicProblem {
    private final LogicManager logicManager;
    private final ProbabilisticSoftLogicObjectiveFunction objectiveFunction;
    private final ImmutableSet<Constraint> constraints;
    private final BiMap<Integer, Integer> externalToInternalIdsMap;
    private final Map<Integer, List<Integer>> internalIdsToFunctionTermsMap;
    private final Map<Integer, CholeskyDecomposition> subProblemCholeskyFactors = new HashMap<>();

    public static final class Builder {
        private final List<FunctionTerm> functionTerms = new ArrayList<>();
        private final List<Constraint> constraints = new ArrayList<>();
        private final List<LogicRule> logicRules = new ArrayList<>();

        private final LogicManager logicManager;
        private final BiMap<Integer, Integer> externalToInternalIdsMap;
        private final Map<Integer, List<Integer>> internalIdsToFunctionTermsMap;

        private int nextInternalId = 0;

        public Builder(LogicManager logicManager) {
            this.logicManager = logicManager;
            externalToInternalIdsMap = HashBiMap.create((int) logicManager.getNumberOfEntityTypes());
            internalIdsToFunctionTermsMap = new HashMap<>((int) logicManager.getNumberOfEntityTypes());
        }

        public int getNumberOfTerms() {
            return functionTerms.size();
        }

        public Builder addLogicRule(LogicRule logicRule) {
            logicRules.add(logicRule);
            return this;
        }

        public Builder addLogicRules(List<LogicRule> logicRules) {
            this.logicRules.addAll(logicRules);
            return this;
        }

        public Builder addGroundPredicate(Predicate predicate, List<Long> argumentAssignments, Double value) {
            logicManager.addGroundPredicate(predicate, argumentAssignments, value);
            return this;
        }

        public Builder addRule(List<GroundPredicate> groundPredicates,
                                boolean[] variableNegations,
                                double power,
                                double weight) {
            List<Integer> internalVariableIds = new ArrayList<>();
            List<Boolean> internalVariableNegations = new ArrayList<>();
            double observedConstant = 0;
            for (int i = 0; i < groundPredicates.size(); ++i) {
                Double observedValue = groundPredicates.get(i).getValue();
                if (observedValue != null) {
                    if (variableNegations[i])
                        observedConstant += observedValue - 1;
                    else
                        observedConstant -= observedValue;
                } else {
                    if (!externalToInternalIdsMap.containsKey((int) groundPredicates.get(i).getId()))
                        externalToInternalIdsMap.put((int) groundPredicates.get(i).getId(), nextInternalId++);
                    internalVariableIds.add(externalToInternalIdsMap.get((int) groundPredicates.get(i).getId()));
                    internalVariableNegations.add(variableNegations[i]);
                }
            }
            double ruleMaximumValue = 1 + observedConstant;
            if (ruleMaximumValue <= 0)
                return this;
            Set<Integer> variableIdsSet = new HashSet<>(internalVariableIds);
            int[] variableIds = ArrayUtils.toPrimitive(variableIdsSet.toArray(new Integer[variableIdsSet.size()]));
            if (variableIds.length == 0)
                return this;
            int indexTerm = functionTerms.size();
            LinearFunction linearFunction = new LinearFunction(Vectors.dense(variableIds.length), ruleMaximumValue);
            for (int variableIndex = 0; variableIndex < internalVariableIds.size(); variableIndex++) {
                if (!internalIdsToFunctionTermsMap.containsKey(internalVariableIds.get(variableIndex)))
                    internalIdsToFunctionTermsMap.put(internalVariableIds.get(variableIndex), new ArrayList<>());
                internalIdsToFunctionTermsMap.get(internalVariableIds.get(variableIndex)).add(indexTerm);
                Vector coefficients = Vectors.dense(variableIds.length);
                if (internalVariableNegations.get(variableIndex)) {
                    coefficients.set(ArrayUtils.indexOf(variableIds, internalVariableIds.get(variableIndex)), 1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, -1));
                } else {
                    coefficients.set(ArrayUtils.indexOf(variableIds, internalVariableIds.get(variableIndex)), -1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, 0));
                }
            }
            functionTerms.add(new FunctionTerm(variableIds, linearFunction, power, weight));
            return this;
        }

//        // BUG BUGBUGBUG: we should handle not adding redundant rules since we handle that on rules
//        public Builder addConstraint(AbstractConstraint constraint, int... externalVariableIndexes) {
//            List<Integer> internalVariableIndexes = new ArrayList<>();
//            for (int externalVariableIndex : externalVariableIndexes) {
//                int internalVariableIndex = externalToInternalIdsMap.getOrDefault(externalVariableIndex, -1);
//                if (internalVariableIndex < 0) {
//                    internalVariableIndex = nextInternalId++;
//                    externalToInternalIdsMap.put(externalVariableIndex, internalVariableIndex);
//                }
//                internalVariableIndexes.add(internalVariableIndex);
//            }
//            constraints.add(new Constraint(constraint, Ints.toArray(internalVariableIndexes)));
//            return this;
//        }

        public ProbabilisticSoftLogicProblem build() {
            return build(true);
        }

        public ProbabilisticSoftLogicProblem build(boolean groundRules) {
            if (groundRules) {
                List<Formula> ruleFormulas = new ArrayList<>();
                for (LogicRule logicRule : logicRules) {
                    List<Formula> disjunctionComponents = new ArrayList<>();
                    for (int i = 0; i < logicRule.bodyFormulas.size(); ++i)
                        if (logicRule.bodyFormulas.get(i) instanceof Negation)
                            disjunctionComponents.add(((Negation) logicRule.bodyFormulas.get(i)).getFormula());
                        else
                            disjunctionComponents.add(new Negation(logicRule.bodyFormulas.get(i)));
                    for (int i = 0; i < logicRule.headFormulas.size(); ++i)
                        disjunctionComponents.add(logicRule.headFormulas.get(i));
                    ruleFormulas.add(new Disjunction(disjunctionComponents));
                }
            InMemoryLazyGrounding grounding = new InMemoryLazyGrounding(logicManager);
//                DatabaseLazyGrounding grounding = new DatabaseLazyGrounding((DatabaseLogicManager) logicManager);
                ruleFormulas = grounding.ground(ruleFormulas);
                for (int ruleIndex = 0; ruleIndex < logicRules.size(); ruleIndex++) {
                    Disjunction ruleFormula = ((Disjunction) ruleFormulas.get(ruleIndex));
                    boolean[] variableNegations = new boolean[ruleFormula.getNumberOfComponents()];
                    for (int i = 0; i < ruleFormula.getNumberOfComponents(); ++i)
                        variableNegations[i] = ruleFormula.getComponent(i) instanceof Negation;
                    if (variableNegations.length != 0)
                        for (List<GroundPredicate> groundedRulePredicates : grounding.getGroundedFormulas().get(ruleIndex))
                            if (Double.isNaN(logicRules.get(ruleIndex).weight))
                                addRule(groundedRulePredicates, variableNegations, 1, 1000);
                            else
                                addRule(groundedRulePredicates,
                                        variableNegations,
                                        logicRules.get(ruleIndex).power,
                                        logicRules.get(ruleIndex).weight);
                }
            }
            return new ProbabilisticSoftLogicProblem(this);
        }

        private static class FunctionTerm {
            private final LinearFunction linearFunction;
            private final int[] variableIndexes;
            private final double power;
            private final double weight;

            private FunctionTerm(int[] variableIndexes,
                                 LinearFunction linearFunction,
                                 double power,
                                 double weight) {
                this.linearFunction = linearFunction;
                this.variableIndexes = variableIndexes;
                this.power = power;
                this.weight = weight;
            }

            @Override
            public String toString() {
                StringBuilder stringBuilder = new StringBuilder();
                stringBuilder
                        .append("Linear Function: ")
                        .append(linearFunction.toString())
                        .append(", Variable Indexes: [");
                for (int variableIndex = 0; variableIndex < variableIndexes.length; ++variableIndex) {
                    if (variableIndex > 0)
                        stringBuilder.append(", ");
                    stringBuilder.append(variableIndexes[variableIndex]);
                }
                stringBuilder.append("], Power: ").append(power);
                return stringBuilder.toString();
            }
        }
    }

    private ProbabilisticSoftLogicProblem(Builder builder) {
        logicManager = builder.logicManager;
        externalToInternalIdsMap = builder.externalToInternalIdsMap;
        SumFunction.Builder sumFunctionBuilder = new SumFunction.Builder(externalToInternalIdsMap.size());
        for (Builder.FunctionTerm function : builder.functionTerms) {
            MaxFunction.Builder maxFunctionBuilder = new MaxFunction.Builder(externalToInternalIdsMap.size());
            maxFunctionBuilder.addConstantTerm(0);
            maxFunctionBuilder.addFunctionTerm(function.linearFunction);
            sumFunctionBuilder.addTerm(
                    new ProbabilisticSoftLogicSumFunctionTerm(maxFunctionBuilder.build(),
                                                              function.power,
                                                              function.weight),
                    function.variableIndexes
            );
        }
        objectiveFunction = new ProbabilisticSoftLogicObjectiveFunction(sumFunctionBuilder);
        constraints = ImmutableSet.copyOf(builder.constraints);
        internalIdsToFunctionTermsMap = builder.internalIdsToFunctionTermsMap;
        for (int subProblemIndex = 0; subProblemIndex < objectiveFunction.getNumberOfTerms(); subProblemIndex++) {
            ProbabilisticSoftLogicSumFunctionTerm objectiveTerm =
                    (ProbabilisticSoftLogicSumFunctionTerm) objectiveFunction.getTerm(subProblemIndex);
            Vector coefficients = objectiveTerm.getLinearFunction().getA();
            if (objectiveTerm.getPower() == 2 && coefficients.size() > 2)
                subProblemCholeskyFactors.put(
                        subProblemIndex,
                        new CholeskyDecomposition(coefficients
                                                          .outer(coefficients)
                                                          .multiply(2 * objectiveTerm.weight)
                                                          .add(Matrix.generateIdentityMatrix(coefficients.size())))
                );
        }
    }

    public List<GroundPredicate> solve() {
        return solve(ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelectionMethod.ALL, null, -1);
    }

    public List<GroundPredicate> solve(
            ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelectionMethod subProblemSelectionMethod,
            ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelector subProblemSelector,
            int numberOfSubProblemSamples) {
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder solverBuilder =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(
                        objectiveFunction,
                        Vectors.dense(objectiveFunction.getNumberOfVariables())
                )
                        .subProblemSolver(
                                (subProblem) -> solveProbabilisticSoftLogicSubProblem(subProblem,
                                                                                      subProblemCholeskyFactors)
                        )
                        .subProblemSelector(subProblemSelector)
                        .subProblemSelectionMethod(subProblemSelectionMethod)
                        .numberOfSubProblemSamples(numberOfSubProblemSamples)
                        .penaltyParameter(1)
                        .penaltyParameterSettingMethod(ConsensusAlternatingDirectionsMethodOfMultipliersSolver
                                        .PenaltyParameterSettingMethod.CONSTANT)
                        .checkForPointConvergence(false)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .logObjectiveValue(false)
                        .logGradientNorm(false)
                        .loggingLevel(3);
        for (Constraint constraint : constraints)
            solverBuilder.addConstraint(constraint.constraint, constraint.variableIndexes);
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver = solverBuilder.build();
        Vector result = solver.solve();
        List<GroundPredicate> groundPredicates = new ArrayList<>();
        for (int internalVariableId = 0; internalVariableId < result.size(); internalVariableId++) {
            GroundPredicate groundPredicate =
                    logicManager.getGroundPredicate(externalToInternalIdsMap.inverse().get(internalVariableId));
            groundPredicate.setValue(result.get(internalVariableId));
            groundPredicates.add(groundPredicate);
        }
        return groundPredicates;
    }

    private static void solveProbabilisticSoftLogicSubProblem(
            ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblem subProblem,
            Map<Integer, CholeskyDecomposition> subProblemCholeskyFactors
    ) {
        ProbabilisticSoftLogicSumFunctionTerm objectiveTerm =
                (ProbabilisticSoftLogicSumFunctionTerm) subProblem.objectiveTerm;
        subProblem.variables.set(
                subProblem.consensusVariables.sub(subProblem.multipliers.div(subProblem.penaltyParameter))
        );
        if (objectiveTerm.getLinearFunction().getValue(subProblem.variables) > 0) {
            if (objectiveTerm.getPower() == 1) {
                subProblem.variables.subInPlace(objectiveTerm.getLinearFunction().getA()
                                                        .mult(objectiveTerm.getWeight() / subProblem.penaltyParameter));
            } else if (objectiveTerm.getPower() == 2) {
                double weight = objectiveTerm.getWeight();
                double constant = objectiveTerm.getLinearFunction().getB();
                subProblem.variables
                        .multInPlace(subProblem.penaltyParameter)
                        .subInPlace(objectiveTerm.getLinearFunction().getA()
                                            .mult(2 * weight * constant));
                if (subProblem.variables.size() == 1) {
                    double coefficient = objectiveTerm.getLinearFunction().getA().get(0);
                    subProblem.variables.divInPlace(2 * weight * coefficient * coefficient
                                                            + subProblem.penaltyParameter);
                } else if (subProblem.variables.size() == 2) {
                    double coefficient0 = objectiveTerm.getLinearFunction().getA().get(0);
                    double coefficient1 = objectiveTerm.getLinearFunction().getA().get(1);
                    double a0 = 2 * weight * coefficient0 * coefficient0 + subProblem.penaltyParameter;
                    double b1 = 2 * weight * coefficient1 * coefficient1 + subProblem.penaltyParameter;
                    double a1b0 = 2 * weight * coefficient0 * coefficient1;
                    subProblem.variables.set(
                            1,
                            (subProblem.variables.get(1) - a1b0 * subProblem.variables.get(0) / a0)
                                    / (b1 - a1b0 * a1b0 / a0)
                    );
                    subProblem.variables.set(
                            0,
                            (subProblem.variables.get(0) - a1b0 * subProblem.variables.get(1)) / a0
                    );
                } else {
                    try {
                        subProblem.variables.set(
                                subProblemCholeskyFactors.get(subProblem.subProblemIndex).solve(subProblem.variables)
                        );
                    } catch (NonSymmetricMatrixException | NonPositiveDefiniteMatrixException e) {
                        System.err.println("Non-positive definite matrix!!!");
                    }
                }
            } else {
                subProblem.variables.set(
                        new NewtonSolver.Builder(
                                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemObjectiveFunction(
                                        objectiveTerm.getSubProblemObjectiveFunction(),
                                        subProblem.consensusVariables,
                                        subProblem.multipliers,
                                        subProblem.penaltyParameter
                                ),
                                subProblem.variables)
                                .lineSearch(new NoLineSearch(1))
                                .maximumNumberOfIterations(1)
                                .build()
                                .solve()
                );
            }
            if (objectiveTerm.getLinearFunction().getValue(subProblem.variables) < 0) {
                subProblem.variables.set(
                        objectiveTerm.getLinearFunction().projectToHyperplane(subProblem.consensusVariables)
                );
            }
        }
    }

    public Map<Integer, Integer> getExternalToInternalIdsMap() {
        return Collections.unmodifiableMap(this.externalToInternalIdsMap);
    }

    public Map<Integer, Integer> getInternalToExternalIdsMap() {
        return Collections.unmodifiableMap(this.externalToInternalIdsMap.inverse());
    }

    public Map<Integer, List<Integer>> getInternalIdsToFunctionTermsMap() {
        return Collections.unmodifiableMap(this.internalIdsToFunctionTermsMap);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        ProbabilisticSoftLogicProblem that = (ProbabilisticSoftLogicProblem) other;

        if (!objectiveFunction.equals(that.objectiveFunction))
            return false;
        if (!constraints.equals(that.constraints))
            return false;
        if (!externalToInternalIdsMap.equals(that.externalToInternalIdsMap))
            return false;
        if (!internalIdsToFunctionTermsMap.equals(that.internalIdsToFunctionTermsMap))
            return false;
        if (!subProblemCholeskyFactors.equals(that.subProblemCholeskyFactors))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = objectiveFunction.hashCode();
        result = 31 * result + constraints.hashCode();
        result = 31 * result + externalToInternalIdsMap.hashCode();
        result = 31 * result + internalIdsToFunctionTermsMap.hashCode();
        result = 31 * result + subProblemCholeskyFactors.hashCode();
        return result;
    }

    private static final class ProbabilisticSoftLogicObjectiveFunction extends SumFunction {
        private ProbabilisticSoftLogicObjectiveFunction(SumFunction.Builder sumFunctionBuilder) {
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
                return a.outer(a).multiply(
                        weight * power * (power - 1) * Math.pow(linearFunction.computeValue(point), power - 2)
                );
            } else {
                return new Matrix(point.size(), point.size());
            }
        }

        @Override
        public boolean equals(Object other) {
            if (this == other)
                return true;
            if (other == null || getClass() != other.getClass())
                return false;

            ProbabilisticSoftLogicSubProblemObjectiveFunction that =
                    (ProbabilisticSoftLogicSubProblemObjectiveFunction) other;

            if (!linearFunction.equals(that.linearFunction))
                return false;
            if (power != that.power)
                return false;
            if (weight != that.weight)
                return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result = linearFunction.hashCode();
            result = 31 * result + Double.valueOf(power).hashCode();
            result = 31 * result + Double.valueOf(weight).hashCode();
            return result;
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

        @Override
        public boolean equals(Object other) {
            if (this == other)
                return true;
            if (other == null || getClass() != other.getClass())
                return false;

            ProbabilisticSoftLogicSumFunctionTerm that = (ProbabilisticSoftLogicSumFunctionTerm) other;

            if (!maxFunction.equals(that.maxFunction))
                return false;
            if (power != that.power)
                return false;
            if (weight != that.weight)
                return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result = maxFunction.hashCode();
            result = 31 * result + Double.valueOf(power).hashCode();
            result = 31 * result + Double.valueOf(weight).hashCode();
            return result;
        }
    }

    private static final class Constraint {
        private final AbstractConstraint constraint;
        private final int[] variableIndexes;

        private Constraint(AbstractConstraint constraint, int[] variableIndexes) {
            this.constraint = constraint;
            this.variableIndexes = variableIndexes;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other)
                return true;
            if (other == null || getClass() != other.getClass())
                return false;

            Constraint that = (Constraint) other;

            if (!constraint.equals(that.constraint))
                return false;
            if (!variableIndexes.equals(that.variableIndexes))
                return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result = constraint.hashCode();
            result = 31 * result + variableIndexes.hashCode();
            return result;
        }
    }

    public static class LogicRule {
        public final List<Formula> bodyFormulas;
        public final List<Formula> headFormulas;
        public final double power;
        public final double weight;

        public LogicRule(List<Formula> bodyFormulas, List<Formula> headFormulas, double power, double weight) {
            this.bodyFormulas = bodyFormulas;
            this.headFormulas = headFormulas;
            this.power = power;
            this.weight = weight;
        }

        @Override
        public String toString() {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append("{");
            if (Double.isNaN(weight))
                stringBuilder.append("constraint");
            else
                stringBuilder.append(weight);
            stringBuilder.append("} ");
            for (int bodyFormulaIndex = 0; bodyFormulaIndex < bodyFormulas.size(); ++bodyFormulaIndex) {
                if (bodyFormulaIndex > 0)
                    stringBuilder.append(" & ");
                stringBuilder.append(bodyFormulas.get(bodyFormulaIndex).toString());
            }
            stringBuilder.append(" >> ");
            for (int headFormulaIndex = 0; headFormulaIndex < headFormulas.size(); ++headFormulaIndex) {
                if (headFormulaIndex > 0)
                    stringBuilder.append(" | ");
                stringBuilder.append(headFormulas.get(headFormulaIndex).toString());
            }
            if (!Double.isNaN(weight))
                stringBuilder.append(" { power: ").append(power).append("}");
            return stringBuilder.toString();
        }
    }
}
