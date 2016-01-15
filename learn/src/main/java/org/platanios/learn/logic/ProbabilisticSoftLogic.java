package org.platanios.learn.logic;

import com.google.common.base.Objects;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableSet;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.logic.formula.Disjunction;
import org.platanios.learn.logic.formula.Formula;
import org.platanios.learn.logic.formula.Negation;
import org.platanios.learn.logic.formula.Predicate;
import org.platanios.learn.logic.grounding.GroundPredicate;
import org.platanios.learn.logic.grounding.GroundingMethod;
import org.platanios.learn.logic.grounding.InMemoryLazyGrounding;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.ConsensusADMMSolver;
import org.platanios.learn.optimization.NewtonSolver;
import org.platanios.learn.optimization.constraint.AbstractConstraint;
import org.platanios.learn.optimization.constraint.LinearEqualityConstraint;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.MaxFunction;
import org.platanios.learn.optimization.function.SumFunction;
import org.platanios.learn.optimization.linesearch.NoLineSearch;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class ProbabilisticSoftLogic {
    private final static Logger logger = LogManager.getFormatterLogger("Probabilistic Soft Logic Problem");

    private final LogicManager logicManager;
    private final SumFunction objectiveFunction;
    private final ImmutableSet<Constraint> constraints;
    private final BiMap<Long, Integer> externalToInternalIdsMap;
    private final Map<Integer, CholeskyDecomposition> subProblemCholeskyFactors = new HashMap<>();

    private ConsensusADMMSolver solver;
    private ConsensusADMMSolver.Builder solverBuilder;

    public static final class Builder {
        private final List<LogicRule> logicRules = new ArrayList<>();
        private final List<FunctionTerm> functionTerms = new ArrayList<>();
        private final Set<Constraint> constraints = new HashSet<>();

        private final LogicManager logicManager;
        private final BiMap<Long, Integer> externalToInternalIdsMap;

        private GroundingMethod groundingMethod = GroundingMethod.IN_MEMORY;

        private AtomicInteger nextInternalId = new AtomicInteger();

        public Builder(LogicManager logicManager) {
            this.logicManager = logicManager;
            externalToInternalIdsMap = HashBiMap.create((int) logicManager.getNumberOfEntityTypes());
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
                    if (!externalToInternalIdsMap.containsKey(groundPredicates.get(i).getId()))
                        externalToInternalIdsMap.put(groundPredicates.get(i).getId(), nextInternalId.getAndIncrement());
                    internalVariableIds.add(externalToInternalIdsMap.get(groundPredicates.get(i).getId()));
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
            LinearFunction linearFunction = new LinearFunction(Vectors.dense(variableIds.length), ruleMaximumValue);
            for (int variableIndex = 0; variableIndex < internalVariableIds.size(); variableIndex++) {
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

        public Builder groundingMethod(GroundingMethod groundingMethod) {
            this.groundingMethod = groundingMethod;
            return this;
        }

        public ProbabilisticSoftLogic build() {
            return build(true);
        }

        public ProbabilisticSoftLogic build(boolean groundRules) {
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
                InMemoryLazyGrounding grounding = groundingMethod.getGrounding(logicManager);
                ruleFormulas = grounding.ground(ruleFormulas);
                for (int ruleIndex = 0; ruleIndex < logicRules.size(); ruleIndex++) {
                    Disjunction ruleFormula = ((Disjunction) ruleFormulas.get(ruleIndex));
                    boolean[] variableNegations = new boolean[ruleFormula.getNumberOfComponents()];
                    for (int i = 0; i < ruleFormula.getNumberOfComponents(); ++i)
                        variableNegations[i] = ruleFormula.getComponent(i) instanceof Negation;
                    if (variableNegations.length != 0)
                        for (List<GroundPredicate> groundedRulePredicates
                                : grounding.getGroundedFormulas().get(ruleIndex))
                            if (Double.isNaN(logicRules.get(ruleIndex).weight))
                                addRule(groundedRulePredicates, variableNegations, 1, 1000);
                            else
                                addRule(groundedRulePredicates,
                                        variableNegations,
                                        logicRules.get(ruleIndex).power,
                                        logicRules.get(ruleIndex).weight);
                }
            }
            return new ProbabilisticSoftLogic(this);
        }

        private static class FunctionTerm {
            private final LinearFunction linearFunction;
            private final int[] variableIndexes;
            private final double power;
            private final double weight;

            private FunctionTerm(int[] variableIndexes, LinearFunction linearFunction, double power, double weight) {
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

    private ProbabilisticSoftLogic(Builder builder) {
        logicManager = builder.logicManager;
        SumFunction.Builder sumFunctionBuilder = new SumFunction.Builder(builder.externalToInternalIdsMap.size());
        for (Builder.FunctionTerm function : builder.functionTerms) {
            MaxFunction maxFunction =
                    new MaxFunction.Builder(builder.externalToInternalIdsMap.size())
                            .addConstantTerm(0).addFunctionTerm(function.linearFunction)
                            .build();
            sumFunctionBuilder.addTerm(
                    new SumFunctionTerm(maxFunction, function.power, function.weight),
                    function.variableIndexes
            );
        }
        objectiveFunction = sumFunctionBuilder.build();
        constraints = ImmutableSet.copyOf(builder.constraints);
        externalToInternalIdsMap = builder.externalToInternalIdsMap;
        for (int subProblemIndex = 0; subProblemIndex < objectiveFunction.getNumberOfTerms(); subProblemIndex++) {
            SumFunctionTerm objectiveTerm =
                    (SumFunctionTerm) objectiveFunction.getTerm(subProblemIndex);
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
        return solve(ConsensusADMMSolver.SubProblemSelectionMethod.ALL, -1);
    }

    public List<GroundPredicate> solve(ConsensusADMMSolver.SubProblemSelector subProblemSelector,
                                       int numberOfSubProblemSamples) {
        if (solverBuilder == null)
            solverBuilder = new ConsensusADMMSolver.Builder(objectiveFunction,
                                                            Vectors.dense(objectiveFunction.getNumberOfVariables()));
        return solve(solverBuilder
                             .subProblemSelector(subProblemSelector)
                             .numberOfSubProblemSamples(numberOfSubProblemSamples));
    }

    public List<GroundPredicate> solve(ConsensusADMMSolver.SubProblemSelectionMethod subProblemSelectionMethod) {
        if (subProblemSelectionMethod != ConsensusADMMSolver.SubProblemSelectionMethod.ALL
                && subProblemSelectionMethod != ConsensusADMMSolver.SubProblemSelectionMethod.CUSTOM)
            throw new IllegalArgumentException("The selected sub-problem selection method cannot be used with this " +
                                                       "solve method, as the number of sub-problem samples is " +
                                                       "required as well.");
        if (solverBuilder == null)
            solverBuilder = new ConsensusADMMSolver.Builder(objectiveFunction,
                                                            Vectors.dense(objectiveFunction.getNumberOfVariables()));
        return solve(solverBuilder.subProblemSelectionMethod(subProblemSelectionMethod));
    }

    public List<GroundPredicate> solve(ConsensusADMMSolver.SubProblemSelectionMethod subProblemSelectionMethod,
                                       int numberOfSubProblemSamples) {
        if (subProblemSelectionMethod == ConsensusADMMSolver.SubProblemSelectionMethod.CUSTOM)
            throw new IllegalArgumentException("The selected sub-problem selection method cannot be used with this " +
                                                       "solve method. The one that takes a sub-problem selector as " +
                                                       "its first argument should be used instead.");
        if (solverBuilder == null)
            solverBuilder = new ConsensusADMMSolver.Builder(objectiveFunction,
                                                            Vectors.dense(objectiveFunction.getNumberOfVariables()));
        return solve(solverBuilder
                             .subProblemSelectionMethod(subProblemSelectionMethod)
                             .numberOfSubProblemSamples(numberOfSubProblemSamples));
    }

    private List<GroundPredicate> solve(ConsensusADMMSolver.Builder solverBuilder) {
        for (Constraint constraint : constraints)
            solverBuilder.addConstraint(constraint.constraint, constraint.variableIndexes);
        solver =
                solverBuilder
                        .subProblemSolver(
                                (subProblem) -> solveProbabilisticSoftLogicSubProblem(subProblem,
                                                                                      subProblemCholeskyFactors)
                        )
                        .penaltyParameterSettingMethod(ConsensusADMMSolver.PenaltyParameterSettingMethod.CONSTANT)
                        .penaltyParameter(1)
                        .checkForPointConvergence(false)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .logObjectiveValue(false)
                        .logGradientNorm(false)
                        .loggingLevel(0)
                        .build();
        Vector result = solver.solve();
        this.solverBuilder = solverBuilder.initialPoint(solver.currentPoint);
        List<GroundPredicate> groundPredicates = new ArrayList<>();
        for (int internalId = 0; internalId < result.size(); internalId++) {
            GroundPredicate groundPredicate =
                    logicManager.getGroundPredicate(externalToInternalIdsMap.inverse().get(internalId));
            groundPredicate.setValue(result.get(internalId));
            groundPredicates.add(groundPredicate);
        }
        return groundPredicates;
    }

    public void fixDataInstanceLabel(GroundPredicate groundPredicate) {
        if (externalToInternalIdsMap.containsKey(groundPredicate.getId()))
            solverBuilder.addConstraint(new LinearEqualityConstraint(Vectors.dense(1, 1), groundPredicate.getValue()),
                                        externalToInternalIdsMap.get(groundPredicate.getId()));
    }

    private static void solveProbabilisticSoftLogicSubProblem(
            ConsensusADMMSolver.SubProblem subProblem,
            Map<Integer, CholeskyDecomposition> subProblemCholeskyFactors
    ) {
        SumFunctionTerm objectiveTerm = (SumFunctionTerm) subProblem.getObjectiveTerm();
        subProblem.getVariables().set(subProblem.getConsensusVariables()
                                              .sub(subProblem.getMultipliers().div(subProblem.getPenaltyParameter())));
        if (objectiveTerm.getLinearFunction().getValue(subProblem.getVariables()) > 0) {
            if (objectiveTerm.getPower() == 1) {
                subProblem.getVariables().subInPlace(
                        objectiveTerm.getLinearFunction().getA()
                                .mult(objectiveTerm.getWeight() / subProblem.getPenaltyParameter())
                );
            } else if (objectiveTerm.getPower() == 2) {
                double weight = objectiveTerm.getWeight();
                double constant = objectiveTerm.getLinearFunction().getB();
                subProblem.getVariables()
                        .multInPlace(subProblem.getPenaltyParameter())
                        .subInPlace(objectiveTerm.getLinearFunction().getA().mult(2 * weight * constant));
                if (subProblem.getVariables().size() == 1) {
                    double coefficient = objectiveTerm.getLinearFunction().getA().get(0);
                    subProblem.getVariables()
                            .divInPlace(2 * weight * coefficient * coefficient + subProblem.getPenaltyParameter());
                } else if (subProblem.getVariables().size() == 2) {
                    double coefficient0 = objectiveTerm.getLinearFunction().getA().get(0);
                    double coefficient1 = objectiveTerm.getLinearFunction().getA().get(1);
                    double a0 = 2 * weight * coefficient0 * coefficient0 + subProblem.getPenaltyParameter();
                    double b1 = 2 * weight * coefficient1 * coefficient1 + subProblem.getPenaltyParameter();
                    double a1b0 = 2 * weight * coefficient0 * coefficient1;
                    subProblem.getVariables().set(
                            1,
                            (subProblem.getVariables().get(1) - a1b0 * subProblem.getVariables().get(0) / a0)
                                    / (b1 - a1b0 * a1b0 / a0)
                    );
                    subProblem.getVariables().set(
                            0,
                            (subProblem.getVariables().get(0) - a1b0 * subProblem.getVariables().get(1)) / a0
                    );
                } else {
                    try {
                        subProblem.getVariables().set(subProblemCholeskyFactors.get(subProblem.getSubProblemIndex())
                                                              .solve(subProblem.getVariables()));
                    } catch (NonSymmetricMatrixException | NonPositiveDefiniteMatrixException e) {
                        logger.error("Non-positive definite matrix encountered while solving a  probabilistic soft " +
                                             "logic sub-problem!");
                    }
                }
            } else {
                subProblem.getVariables().set(
                        new NewtonSolver.Builder(
                                new ConsensusADMMSolver.SubProblemObjectiveFunction(
                                        objectiveTerm.getSubProblemObjectiveFunction(),
                                        subProblem.getConsensusVariables(),
                                        subProblem.getMultipliers(),
                                        subProblem.getPenaltyParameter()
                                ),
                                subProblem.getVariables()
                        )
                                .lineSearch(new NoLineSearch(1))
                                .maximumNumberOfIterations(1)
                                .build()
                                .solve()
                );
            }
            if (objectiveTerm.getLinearFunction().getValue(subProblem.getVariables()) < 0) {
                subProblem.getVariables().set(
                        objectiveTerm.getLinearFunction().projectToHyperplane(subProblem.getConsensusVariables())
                );
            }
        }
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        ProbabilisticSoftLogic that = (ProbabilisticSoftLogic) other;

        return Objects.equal(objectiveFunction, that.objectiveFunction)
                && Objects.equal(constraints, that.constraints)
                && Objects.equal(externalToInternalIdsMap, that.externalToInternalIdsMap)
                && Objects.equal(subProblemCholeskyFactors, that.subProblemCholeskyFactors);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(objectiveFunction,
                                constraints,
                                externalToInternalIdsMap,
                                subProblemCholeskyFactors);
    }

    private static final class SubProblemObjectiveFunction extends AbstractFunction {
        private final LinearFunction linearFunction;
        private final double power;
        private final double weight;

        private SubProblemObjectiveFunction(LinearFunction linearFunction, double power, double weight) {
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
            if (power > 0)
                return linearFunction.computeGradient(point).mult(
                        weight * power * Math.pow(linearFunction.computeValue(point), power - 1)
                );
            else
                return Vectors.build(point.size(), point.type());
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

            SubProblemObjectiveFunction that = (SubProblemObjectiveFunction) other;

            return Objects.equal(linearFunction, that.linearFunction)
                    && Objects.equal(power, that.power)
                    && Objects.equal(weight, that.weight);
        }

        @Override
        public int hashCode() {
            return Objects.hashCode(linearFunction, power, weight);
        }
    }

    private static final class SumFunctionTerm extends AbstractFunction {
        private final MaxFunction maxFunction;
        private final double power;
        private final double weight;

        private SumFunctionTerm(MaxFunction maxFunction, double power, double weight) {
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

        private SubProblemObjectiveFunction getSubProblemObjectiveFunction() {
            return new SubProblemObjectiveFunction(
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

            SumFunctionTerm that = (SumFunctionTerm) other;

            return Objects.equal(maxFunction, that.maxFunction)
                    && Objects.equal(power, that.power)
                    && Objects.equal(weight, that.weight);
        }

        @Override
        public int hashCode() {
            return Objects.hashCode(maxFunction, power, weight);
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

            return Objects.equal(constraint, that.constraint) && Objects.equal(variableIndexes, that.variableIndexes);
        }

        @Override
        public int hashCode() {
            return Objects.hashCode(constraint, variableIndexes);
        }
    }

    public static class LogicRule {
        private final List<Formula> bodyFormulas;
        private final List<Formula> headFormulas;
        private final Double power;
        private final Double weight;

        public LogicRule(List<Formula> bodyFormulas, List<Formula> headFormulas) {
            this.bodyFormulas = bodyFormulas;
            this.headFormulas = headFormulas;
            power = null;
            weight = null;
        }

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
            if (weight == null)
                stringBuilder.append("constraint");
            else
                stringBuilder.append(weight);
            stringBuilder.append("} ");
            for (int bodyFormulaIndex = 0; bodyFormulaIndex < bodyFormulas.size(); bodyFormulaIndex++) {
                if (bodyFormulaIndex > 0)
                    stringBuilder.append(" & ");
                stringBuilder.append(bodyFormulas.get(bodyFormulaIndex).toString());
            }
            stringBuilder.append(" -> ");
            for (int headFormulaIndex = 0; headFormulaIndex < headFormulas.size(); headFormulaIndex++) {
                if (headFormulaIndex > 0)
                    stringBuilder.append(" | ");
                stringBuilder.append(headFormulas.get(headFormulaIndex).toString());
            }
            if (weight != null)
                stringBuilder.append(" { power: ").append(power).append("}");
            return stringBuilder.toString();
        }
    }
}
