package org.platanios.experiment.psl;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.primitives.Booleans;
import com.google.common.primitives.Ints;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.Utilities;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.Disjunction;
import org.platanios.learn.logic.formula.Formula;
import org.platanios.learn.logic.formula.Negation;
import org.platanios.learn.logic.grounding.LazyGrounding;
import org.platanios.learn.logic.grounding.GroundedPredicate;
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
public final class FastProbabilisticSoftLogicProblem {
    private final BiMap<Integer, Integer> externalToInternalIndexesMapping;
    private final ProbabilisticSoftLogicFunction objectiveFunction;
    private final ImmutableSet<Constraint> constraints;
    private final Map<Integer, CholeskyDecomposition> subProblemCholeskyFactors = new HashMap<>();
    private final Map<Integer, List<Integer>> externalPredicateIdToTerms;

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof FastProbabilisticSoftLogicProblem)) {
            return false;
        }
        if (other == this) {
            return true;
        }

        FastProbabilisticSoftLogicProblem rhs = (FastProbabilisticSoftLogicProblem) other;

        return new EqualsBuilder()
                .append(this.externalToInternalIndexesMapping, rhs.externalToInternalIndexesMapping)
                .append(this.objectiveFunction, rhs.objectiveFunction)
                .append(this.constraints, rhs.constraints)
                .append(this.subProblemCholeskyFactors, rhs.subProblemCholeskyFactors)
                .append(this.externalPredicateIdToTerms, rhs.externalPredicateIdToTerms)
                .isEquals();

    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 31)
                .append(this.externalToInternalIndexesMapping)
                .append(this.objectiveFunction)
                .append(this.constraints)
                .append(this.subProblemCholeskyFactors)
                .append(this.externalPredicateIdToTerms)
                .toHashCode();
    }

    public static class Rule {

        public Rule(List<Formula<Integer>> bodyParts, List<Formula<Integer>> headParts, double weight, double power) {
            this.bodyParts = bodyParts;
            this.headParts = headParts;
            this.weight = weight;
            this.power = power;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            if (Double.isNaN(this.weight)) {
                sb.append("constraint");
            } else {
                sb.append(this.weight);
            }
            sb.append("} ");
            for (int i = 0; i < bodyParts.size(); ++i) {
                if (i > 0) {
                    sb.append(" & ");
                }
                sb.append(bodyParts.get(i).toString());
            }
            sb.append(" >> ");
            for (int i = 0; i < headParts.size(); ++i) {
                if (i > 0) {
                    sb.append(" | ");
                }
                sb.append(headParts.get(i).toString());
            }
            if (!Double.isNaN(this.weight)) {
                sb.append(" {");
                if (this.power == 2) {
                    sb.append("squared");
                } else {
                    sb.append(this.power);
                }
                sb.append("}");
            }
            return sb.toString();
        }

        public static void addGroundingsToBuilder(
                List<Rule> rules,
                Builder builder,
                LogicManager<Integer, Double> logicManager) {
            Set<LazyGrounding<Integer, Double>.ActivatedGroundedPredicate<Integer>> activatedGroundedPredicates = new HashSet<>();
            int previousTotalNumberOfRules = 0;
            int currentTotalNumberOfRules = -1;
            Map<Rule, List<List<GroundedPredicate<Integer, Double>>>> groundedRules = new HashMap<>();
            while (previousTotalNumberOfRules != currentTotalNumberOfRules) {
                previousTotalNumberOfRules = currentTotalNumberOfRules;
                currentTotalNumberOfRules = 0;
                for (Rule rule : rules) {
                    List<Formula<Integer>> disjunctionComponents = new ArrayList<>();
                    boolean[] bodyNegations = new boolean[rule.bodyParts.size()];
                    for (int i = 0; i < rule.bodyParts.size(); ++i) {
                        bodyNegations[i] = rule.bodyParts.get(i) instanceof Negation;
                        disjunctionComponents.add(new Negation<>(rule.bodyParts.get(i)));
                    }
                    boolean[] headNegations = new boolean[rule.headParts.size()];
                    for (int i = 0; i < rule.headParts.size(); ++i) {
                        headNegations[i] = rule.headParts.get(i) instanceof Negation;
                        disjunctionComponents.add(rule.headParts.get(i));
                    }
                    Formula<Integer> ruleFormula = new Disjunction<>(disjunctionComponents);
                    LazyGrounding<Integer, Double> exhaustiveGrounding = new LazyGrounding<>(logicManager, activatedGroundedPredicates);
                    exhaustiveGrounding.ground(ruleFormula);
                    Set<LazyGrounding<Integer, Double>.ActivatedGroundedPredicate<Integer>> newActivatedGroundedPredicates = exhaustiveGrounding.getActivatedGroundedPredicates();
                    groundedRules.put(rule, exhaustiveGrounding.getGroundedPredicates());
                    currentTotalNumberOfRules += groundedRules.get(rule).size();
                    System.out.println("Generated " + groundedRules.get(rule).size() + " groundings for rule " + rule.toString());
                    activatedGroundedPredicates.addAll(newActivatedGroundedPredicates);
                }
            }
            for (Rule rule : rules) {
                List<List<GroundedPredicate<Integer, Double>>> predicateGroundings = groundedRules.get(rule);
                boolean[] bodyNegations = new boolean[rule.bodyParts.size()];
                for (int i = 0; i < rule.bodyParts.size(); ++i)
                    bodyNegations[i] = rule.bodyParts.get(i) instanceof Negation;
                boolean[] headNegations = new boolean[rule.headParts.size()];
                for (int i = 0; i < rule.headParts.size(); ++i)
                    headNegations[i] = rule.headParts.get(i) instanceof Negation;
                for (List<GroundedPredicate<Integer, Double>> groundedRulePredicates : predicateGroundings) {
                    if (rule.bodyParts.size() + rule.headParts.size() == 0)
                        continue;
                    int[] bodyVariableIndexes = new int[rule.bodyParts.size()];
                    for (int i = 0; i < rule.bodyParts.size(); ++i)
                        bodyVariableIndexes[i] = (int) groundedRulePredicates.get(i).getIdentifier();
                    int[] headVariableIndexes = new int[rule.headParts.size()];
                    for (int i = 0; i < rule.headParts.size(); ++i)
                        headVariableIndexes[i] = (int) groundedRulePredicates.get(rule.bodyParts.size() + i).getIdentifier();
                    if (Double.isNaN(rule.weight))
                        builder.addRule(headVariableIndexes, bodyVariableIndexes, headNegations, bodyNegations, 1, 1000);
                    else
                        builder.addRule(headVariableIndexes, bodyVariableIndexes, headNegations, bodyNegations, rule.power, rule.weight);
                }
            }
        }

        // NaN indicates constraint
        public final List<Formula<Integer>> bodyParts;
        public final List<Formula<Integer>> headParts;
        public final double weight;
        public final double power;
    }

    public static final class Builder {
        private final BiMap<Integer, Integer> externalToInternalIndexesMapping;

        private final Map<Integer, Double> observedVariableValues;
        //private final HashMap<String, FunctionTerm> functionTerms = new HashMap<>();
        private final List<FunctionTerm> functionTerms = new ArrayList<>();
        private final List<Constraint> constraints = new ArrayList<>();
        private final Map<Integer, List<Integer>> externalPredicateIdToTerms;

        private int nextInternalIndex = 0;

        public Builder(
                int[] observedVariableIndexes,
                double[] observedVariableValues,
                int numberOfUnobservedVariables) {

            ImmutableMap.Builder<Integer, Double> observedVariableValueBuilder = ImmutableMap.builder();
            if ((observedVariableIndexes == null) != (observedVariableValues == null)) {
                throw new IllegalArgumentException(
                        "The provided indexes for the observed variables must much the corresponding provided values."
                );
            }
            if (observedVariableIndexes != null) {
                if (observedVariableIndexes.length != observedVariableValues.length) {
                    throw new IllegalArgumentException(
                            "The provided indexes array for the observed variables must " +
                                    "have the same length the corresponding provided values array."
                    );
                }
                for (int i = 0; i < observedVariableIndexes.length; ++i) {
                    observedVariableValueBuilder.put(observedVariableIndexes[i], observedVariableValues[i]);
                }
            }

            // special case - always add -1 as an Id with observed value 0.
            // predicates which are grounded outside of a closed set will thus have value 0
//            observedVariableValueBuilder.put(-1, 0.0);
            this.observedVariableValues = observedVariableValueBuilder.build();
            this.externalToInternalIndexesMapping = HashBiMap.create(numberOfUnobservedVariables);
            this.externalPredicateIdToTerms = new HashMap<>(numberOfUnobservedVariables + observedVariableIndexes.length, 1);
        }

        public int getNumberOfTerms() { return this.functionTerms.size(); }

        Builder addRule(
                int[] headVariableIndexes,
                int[] bodyVariableIndexes,
                boolean[] headNegations,
                boolean[] bodyNegations,
                double power,
                double weight) {
            RulePart headPart = convertRulePartToInternalRepresentation(headVariableIndexes, headNegations, true);
            RulePart bodyPart = convertRulePartToInternalRepresentation(bodyVariableIndexes, bodyNegations, false);
            double ruleMaximumValue = 1 + headPart.observedConstant + bodyPart.observedConstant;
            if (ruleMaximumValue <= 0)
                return this;
            int[] variableIndexes = Utilities.union(headPart.variableIndexes, bodyPart.variableIndexes);
            if (variableIndexes.length == 0)
                return this;
            int indexTerm = this.functionTerms.size();
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
                List<Integer> predicateTermIndices = this.externalPredicateIdToTerms.getOrDefault(bodyPart.variableIndexes[bodyVariable], null);
                if (predicateTermIndices == null) {
                    predicateTermIndices = new ArrayList<>(200);
                    this.externalPredicateIdToTerms.put(bodyPart.variableIndexes[bodyVariable], predicateTermIndices);
                }
                predicateTermIndices.add(indexTerm);
                Vector coefficients = Vectors.dense(variableIndexes.length);
                if (bodyPart.negations[bodyVariable]) {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, bodyPart.variableIndexes[bodyVariable]), -1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, 0));
                } else {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, bodyPart.variableIndexes[bodyVariable]), 1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, -1));
                }
            }

            for (int headVariable = 0; headVariable < headVariableIndexes.length; ++headVariable) {
                List<Integer> predicateTermIndices = this.externalPredicateIdToTerms.getOrDefault(headVariableIndexes[headVariable], null);
                if (predicateTermIndices == null) {
                    predicateTermIndices = new ArrayList<>(200);
                    this.externalPredicateIdToTerms.put(headVariableIndexes[headVariable], predicateTermIndices);
                }
                predicateTermIndices.add(indexTerm);
            }
            for (int bodyVariable = 0; bodyVariable < bodyVariableIndexes.length; ++bodyVariable) {
                List<Integer> predicateTermIndices = this.externalPredicateIdToTerms.getOrDefault(bodyVariableIndexes[bodyVariable], null);
                if (predicateTermIndices == null) {
                    predicateTermIndices = new ArrayList<>(200);
                    this.externalPredicateIdToTerms.put(bodyVariableIndexes[bodyVariable], predicateTermIndices);
                }
                predicateTermIndices.add(indexTerm);
            }

            FunctionTerm term = new FunctionTerm(variableIndexes, linearFunction, weight, power);
            // functionTerms.putIfAbsent(term.toString(), term);
            functionTerms.add(term);
            return this;
        }

        // BUG BUGBUGBUG: we should handle not adding redundant rules since we handle that on rules
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

        public FastProbabilisticSoftLogicProblem build() {
            return new FastProbabilisticSoftLogicProblem(this);
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

            @Override
            public String toString() {
                StringBuilder sb = new StringBuilder();
                sb.append("linFunc_a:[");
                Vector a = this.linearFunction.getA();
                for (int iComp = 0; iComp < a.size(); ++iComp) {
                    if (iComp > 0) {
                        sb.append(",");
                    }
                    sb.append(a.get(iComp));
                }
                sb.append("];linFunc_b:");
                sb.append(this.linearFunction.getB());
                sb.append(";idx:");
                for (int iIdx = 0; iIdx < this.variableIndexes.length; ++iIdx) {
                    if (iIdx > 0) {
                        sb.append(",");
                    }
                    sb.append(this.variableIndexes[iIdx]);
                }
                sb.append(";pwr:");
                sb.append(this.power);
                return sb.toString();
            }
        }
    }

    private FastProbabilisticSoftLogicProblem(Builder builder) {
        this.externalToInternalIndexesMapping = builder.externalToInternalIndexesMapping;
        SumFunction.Builder sumFunctionBuilder = new SumFunction.Builder(externalToInternalIndexesMapping.size());
        for (Builder.FunctionTerm function : builder.functionTerms) {
            // for (Builder.FunctionTerm function : builder.functionTerms.values()) {
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
        this.objectiveFunction = new ProbabilisticSoftLogicFunction(sumFunctionBuilder);
        this.constraints = ImmutableSet.copyOf(builder.constraints);

        this.externalPredicateIdToTerms = builder.externalPredicateIdToTerms;

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

    public Map<Integer, Double> solve() {
        return this.solve(ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelectionMethod.ALL, null, -1);
    }

    public Map<Integer, Double> solve(
            ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelectionMethod subProblemSelectionMethod,
            ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelector subProblemSelector,
            int numberOfSubProblemSamples) {
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder solverBuilder =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(
                        objectiveFunction,
                        Vectors.dense(objectiveFunction.getNumberOfVariables())
                )
                        .subProblemSolver((subProblem) -> solveProbabilisticSoftLogicSubProblem(subProblem, subProblemCholeskyFactors))
                        .subProblemSelector(subProblemSelector)
                        .subProblemSelectionMethod(subProblemSelectionMethod) // if this is not CUSTOM, it will override the subProblemSelector
                        .numberOfSubProblemSamples(numberOfSubProblemSamples)
                        .penaltyParameter(1)
                        .penaltyParameterSettingMethod(ConsensusAlternatingDirectionsMethodOfMultipliersSolver.PenaltyParameterSettingMethod.CONSTANT)
                        .checkForPointConvergence(false)
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
                        subProblem.variables.set(subProblemCholeskyFactors.get(subProblem.subProblemIndex).solve(subProblem.variables));
                    } catch (NonSymmetricMatrixException|NonPositiveDefiniteMatrixException e) {
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

    public Map<Integer, Integer> getExternalToInternalIds() {
        return Collections.unmodifiableMap(this.externalToInternalIndexesMapping);
    }

    public Map<Integer, Integer> getInternalToExternalIds() {
        return Collections.unmodifiableMap(this.externalToInternalIndexesMapping.inverse());
    }

    public Map<Integer, List<Integer>> getExternalPredicateIdsToTerms() {
        return Collections.unmodifiableMap(this.externalPredicateIdToTerms);
    }

    public RandomWalkSampler.TermPredicateIdGetter getTermPredicateIdGetter() {
        return new RandomWalkSampler.TermPredicateIdGetter() {
            @Override
            public int[] getInternalPredicateIds(int term) {
                return FastProbabilisticSoftLogicProblem.this.objectiveFunction.getTermIndices(term);
            }
        };
    }

    private static final class ProbabilisticSoftLogicFunction extends SumFunction {
        private ProbabilisticSoftLogicFunction(SumFunction.Builder sumFunctionBuilder) {
            super(sumFunctionBuilder);
        }

        // dangerous, should return unmodifiable collection
        public int[] getTermIndices(int term) {
            return this.termsVariables.get(term);
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
        public boolean equals(Object other) {

            if (other == this) {
                return true;
            }

            if (!super.equals(other)) {
                return false;
            }

            if (!(other instanceof ProbabilisticSoftLogicSubProblemObjectiveFunction)) {
                return false;
            }

            ProbabilisticSoftLogicSubProblemObjectiveFunction rhs = (ProbabilisticSoftLogicSubProblemObjectiveFunction)other;

            return new EqualsBuilder()
                    .append(this.linearFunction, rhs.linearFunction)
                    .append(this.power, rhs.power)
                    .append(this.weight, rhs.weight)
                    .isEquals();

        }

        @Override
        public int hashCode() {

            return new HashCodeBuilder(53, 31)
                    .append(super.hashCode())
                    .append(this.linearFunction)
                    .append(this.power)
                    .append(this.weight)
                    .toHashCode();

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
        public boolean equals(Object other) {

            if (other == this) {
                return true;
            }

            if (!super.equals(other)) {
                return false;
            }

            if (!(other instanceof ProbabilisticSoftLogicSumFunctionTerm)) {
                return false;
            }

            ProbabilisticSoftLogicSumFunctionTerm rhs = (ProbabilisticSoftLogicSumFunctionTerm)other;

            return new EqualsBuilder()
                    .append(this.maxFunction, rhs.maxFunction)
                    .append(this.power, rhs.power)
                    .append(this.weight, rhs.weight)
                    .isEquals();

        }

        @Override
        public int hashCode() {

            return new HashCodeBuilder(67, 17)
                    .append(super.hashCode())
                    .append(this.maxFunction)
                    .append(this.power)
                    .append(this.weight)
                    .toHashCode();

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

        @Override
        public boolean equals(Object other) {
            if(!(other instanceof Constraint)) {
                return false;
            }

            if (other == this) {
                return true;
            }

            Constraint rhs = (Constraint) other;
            return new EqualsBuilder()
                    .append(this.constraint, rhs.constraint)
                    .append(this.variableIndexes, rhs.variableIndexes)
                    .isEquals();
        }

        @Override
        public int hashCode() {
            return new HashCodeBuilder(13, 37)
                    .append(this.constraint)
                    .append(this.variableIndexes)
                    .toHashCode();
        }

    }
}