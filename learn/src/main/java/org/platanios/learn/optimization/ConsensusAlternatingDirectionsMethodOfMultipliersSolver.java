package org.platanios.learn.optimization;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.statistics.StatisticsUtilities;
import org.platanios.learn.optimization.constraint.AbstractConstraint;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.NonSmoothFunctionException;
import org.platanios.learn.optimization.function.SumFunction;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.function.Consumer;

/**
 * TODO: The current implementation is PSL-specific and not generic at all.
 * TODO: Only support p = 1 in PSL for now and only linear equality constraints.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class ConsensusAlternatingDirectionsMethodOfMultipliersSolver extends AbstractIterativeSolver {
    private final Object lock = new Object();
    private final List<Vector> variableCopies = new ArrayList<>();
    private final List<Vector> lagrangeMultipliers = new ArrayList<>();
    private final Vector variableCopiesSum = Vectors.build(currentPoint.size(), currentPoint.type());

    private final Vector variableCopiesCounts;
    private final int maximumNumberOfIterationsWithNoPointChange;
    private final double absoluteTolerance;
    private final double relativeTolerance;
    private final boolean checkForPrimalAndDualResidualConvergence;

    private final PenaltyParameterSettingMethod penaltyParameterSettingMethod;
    private final SubProblemSelectionMethod subProblemSelectionMethod;
    public final int numberOfSubProblemSamples;
    private final double mu;
    private final double tauIncrement;
    private final double tauDecrement;
    private final Consumer<SubProblem> subProblemSolver;
    private final ExecutorService taskExecutor;

    private final SumFunction objective;
    private final List<int[]> constraintsVariablesIndexes;
    private final List<AbstractConstraint> constraints;

    private Vector primalResidualSquaredTerms;
    private double penaltyParameter;
    private double primalResidual;
    private double dualResidual;
    private double primalTolerance;
    private double dualTolerance;

    private int numberOfIterationsWithNoPointChange = 0;
    private boolean primalResidualConverged = false;
    private boolean dualResidualConverged = false;
    private DoubleAdder primalToleranceAdder = new DoubleAdder();
    private DoubleAdder dualToleranceAdder = new DoubleAdder();

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractIterativeSolver.AbstractBuilder<T> {
        protected final List<int[]> constraintsVariablesIndexes = new ArrayList<>();
        protected final List<AbstractConstraint> constraints = new ArrayList<>();

        protected int maximumNumberOfIterationsWithNoPointChange = 1;
        protected double absoluteTolerance = 1e-5;
        protected double relativeTolerance = 1e-3;
        protected boolean checkForPrimalAndDualResidualConvergence = true;

        protected PenaltyParameterSettingMethod penaltyParameterSettingMethod = PenaltyParameterSettingMethod.ADAPTIVE;
        protected SubProblemSelectionMethod subProblemSelectionMethod = SubProblemSelectionMethod.ALL;
        protected int numberOfSubProblemSamples = -1;
        protected double mu = 10;
        protected double tauIncrement = 2;
        protected double tauDecrement = 2;
        protected double penaltyParameter = 1e-4;
        protected Consumer<SubProblem> subProblemSolver = null;
        protected int numberOfThreads = Runtime.getRuntime().availableProcessors();

        protected AbstractBuilder(SumFunction objective,
                                  Vector initialPoint) {
            super(objective, initialPoint);
            checkForPointConvergence(false);
            checkForObjectiveConvergence(false);
            checkForGradientConvergence(false);
        }

        public T addConstraint(AbstractConstraint constraint, int... variableIndexes) {
            constraintsVariablesIndexes.add(variableIndexes);
            constraints.add(constraint);
            return self();
        }

        public T maximumNumberOfIterationsWithNoPointChange(int maximumNumberOfIterationsWithNoPointChange) {
            this.maximumNumberOfIterationsWithNoPointChange = maximumNumberOfIterationsWithNoPointChange;
            return self();
        }

        public T absoluteTolerance(double absoluteTolerance) {
            this.absoluteTolerance = absoluteTolerance;
            return self();
        }

        public T relativeTolerance(double relativeTolerance) {
            this.relativeTolerance = relativeTolerance;
            return self();
        }

        public T checkForPrimalAndDualResidualConvergence(boolean checkForPrimalAndDualResidualConvergence) {
            this.checkForPrimalAndDualResidualConvergence = checkForPrimalAndDualResidualConvergence;
            return self();
        }

        public T penaltyParameterSettingMethod(PenaltyParameterSettingMethod penaltyParameterSettingMethod) {
            this.penaltyParameterSettingMethod = penaltyParameterSettingMethod;
            return self();
        }

        public T subProblemSelectionMethod(SubProblemSelectionMethod subProblemSelectionMethod) {
            this.subProblemSelectionMethod = subProblemSelectionMethod;
            return self();
        }

        /**
         * Note that this parameter is not used if the sub-problem selection method is set to
         * {@link SubProblemSelectionMethod#ALL} (which is the default setting). If a high sub-sampling ratio is used,
         * it is suggested that the value of {@link this#mu} be higher.
         *
         * @param   numberOfSubProblemSamples
         * @return
         */
        public T numberOfSubProblemSamples(int numberOfSubProblemSamples) {
            this.numberOfSubProblemSamples = numberOfSubProblemSamples;
            return self();
        }

        public T mu(double mu) {
            this.mu = mu;
            return self();
        }

        public T tauIncrement(double tauIncrement) {
            this.tauIncrement = tauIncrement;
            return self();
        }

        public T tauDecrement(double tauDecrement) {
            this.tauDecrement = tauDecrement;
            return self();
        }

        public T penaltyParameter(double penaltyParameter) {
            this.penaltyParameter = penaltyParameter;
            return self();
        }

        public T subProblemSolver(Consumer<SubProblem> subProblemSolver) {
            this.subProblemSolver = subProblemSolver;
            return self();
        }

        public T numberOfThreads(int numberOfThreads) {
            this.numberOfThreads = numberOfThreads;
            return self();
        }

        public ConsensusAlternatingDirectionsMethodOfMultipliersSolver build() {
            return new ConsensusAlternatingDirectionsMethodOfMultipliersSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(SumFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private ConsensusAlternatingDirectionsMethodOfMultipliersSolver(AbstractBuilder<?> builder) {
        super(builder);
        objective = (SumFunction) builder.objective;
        constraintsVariablesIndexes = builder.constraintsVariablesIndexes;
        constraints = builder.constraints;
        maximumNumberOfIterationsWithNoPointChange = builder.maximumNumberOfIterationsWithNoPointChange;
        absoluteTolerance = Math.sqrt(currentPoint.size()) * builder.absoluteTolerance;
        relativeTolerance = builder.relativeTolerance;
        checkForPrimalAndDualResidualConvergence = builder.checkForPrimalAndDualResidualConvergence;
        primalResidualSquaredTerms = Vectors.build(objective.getNumberOfTerms(), VectorType.DENSE);
        penaltyParameterSettingMethod = builder.penaltyParameterSettingMethod;
        subProblemSelectionMethod = builder.subProblemSelectionMethod;
        numberOfSubProblemSamples = builder.numberOfSubProblemSamples;
        mu = builder.mu;
        tauIncrement = builder.tauIncrement;
        tauDecrement = builder.tauDecrement;
        penaltyParameter = builder.penaltyParameter;
        subProblemSolver = builder.subProblemSolver;
        taskExecutor = Executors.newFixedThreadPool(builder.numberOfThreads);
        variableCopiesCounts = Vectors.dense(currentPoint.size());
        for (int[] variableIndexes : objective.getTermsVariables()) {
            Vector termPoint = Vectors.build(variableIndexes.length, currentPoint.type());
            termPoint.set(currentPoint.get(variableIndexes));
            variableCopies.add(termPoint);
            lagrangeMultipliers.add(Vectors.build(variableIndexes.length, currentPoint.type()));
            for (int variableIndex : variableIndexes)
                variableCopiesCounts.set(variableIndex, variableCopiesCounts.get(variableIndex) + 1);
        }
        for (int[] variableIndexes : constraintsVariablesIndexes) {
            Vector termPoint = Vectors.build(variableIndexes.length, currentPoint.type());
            termPoint.set(currentPoint.get(variableIndexes));
            variableCopies.add(termPoint);
            lagrangeMultipliers.add(Vectors.build(variableIndexes.length, currentPoint.type()));
            for (int variableIndex : variableIndexes)
                variableCopiesCounts.set(variableIndex, variableCopiesCounts.get(variableIndex) + 1);
        }
    }

    @Override
    public boolean checkTerminationConditions() {
        if (super.checkTerminationConditions()) {
            if (checkForPointConvergence && pointConverged) {
                if ((checkForObjectiveConvergence && objectiveConverged)
                        || (checkForGradientConvergence && gradientConverged)) {
                    return true;
                } else {
                    numberOfIterationsWithNoPointChange++;
                    if (numberOfIterationsWithNoPointChange < maximumNumberOfIterationsWithNoPointChange)
                        pointConverged = false;
                    else
                        return true;
                }
            } else if ((checkForObjectiveConvergence && objectiveConverged)
                    || (checkForGradientConvergence && gradientConverged)) {
                return true;
            } else {
                numberOfIterationsWithNoPointChange = 0;
            }
        }
        if (currentIteration > 0 && checkForPrimalAndDualResidualConvergence) {
            primalTolerance = absoluteTolerance + relativeTolerance
                    * Math.max(Math.sqrt(primalToleranceAdder.doubleValue()),
                               Math.sqrt(currentPoint
                                                 .map(x -> x * x)
                                                 .multElementwise(variableCopiesCounts)
                                                 .sum()));
            dualTolerance = absoluteTolerance + relativeTolerance * Math.sqrt(dualToleranceAdder.doubleValue());
            primalResidualConverged = primalResidual <= primalTolerance;
            dualResidualConverged = dualResidual <= dualTolerance;
            return primalResidualConverged && dualResidualConverged;
        } else {
            return false;
        }
    }

    @Override
    public void printIteration() {
        StringBuilder stringBuilder = new StringBuilder(String.format("Iteration #: %10d", currentIteration));
        if (logObjectiveValue)
            stringBuilder.append(String.format(" | Objective Value: %20s | Objective Change: %20s",
                                               DECIMAL_FORMAT.format(currentObjectiveValue),
                                               DECIMAL_FORMAT.format(objectiveChange)));
        if (checkForPointConvergence)
            stringBuilder.append(String.format(" | Point Change: %20s",
                                               DECIMAL_FORMAT.format(pointChange)));
        if (logGradientNorm)
            stringBuilder.append(String.format(" | Gradient Norm: %20s",
                                               DECIMAL_FORMAT.format(gradientNorm)));
        if (checkForPrimalAndDualResidualConvergence)
            stringBuilder.append(String.format(" | Primal Residual: %20s | Dual Residual: %20s",
                                               DECIMAL_FORMAT.format(primalResidual),
                                               DECIMAL_FORMAT.format(dualResidual)));
        logger.info(stringBuilder.toString());
    }

    @Override
    public void printTerminationMessage() {
        super.printTerminationMessage();
        if (primalResidualConverged)
            logger.info("The L2 norm of the primal residual, %s, was below the convergence threshold of %s.",
                        DECIMAL_FORMAT.format(primalResidual),
                        DECIMAL_FORMAT.format(primalTolerance));
        if (dualResidualConverged)
            logger.info("The L2 norm of the dual residual, %s, was below the convergence threshold of %s.",
                        DECIMAL_FORMAT.format(dualResidual),
                        DECIMAL_FORMAT.format(dualTolerance));
    }

    @Override
    public void performIterationUpdates() {
        previousPoint = currentPoint.copy();
        if (subProblemSelectionMethod == SubProblemSelectionMethod.ALL) {
            primalToleranceAdder = new DoubleAdder();
            dualToleranceAdder = new DoubleAdder();
        }
        int[] selectedSubProblemIndexes = subProblemSelectionMethod.selectSubProblems(this);
        Arrays.sort(selectedSubProblemIndexes);
        Set<Integer> affectedConsensusVariables = new HashSet<>();
        List<Callable<Object>> subProblemTasks = new ArrayList<>();
        List<Callable<Object>> residualComputationTasks = new ArrayList<>();
        int temporaryIndex = 0;
        for (int subProblemIndex = 0; subProblemIndex < objective.getNumberOfTerms(); subProblemIndex++) {
            final boolean solveSubProblem = (temporaryIndex < selectedSubProblemIndexes.length
                    && selectedSubProblemIndexes[temporaryIndex] == subProblemIndex)
                    || currentIteration == 1;
            if (solveSubProblem
                    || penaltyParameterSettingMethod == PenaltyParameterSettingMethod.ADAPTIVE
                    || checkForPrimalAndDualResidualConvergence
                    || subProblemSelectionMethod == SubProblemSelectionMethod.CONSENSUS_FOCUSED_SAMPLING) {
                int[] variableIndexes = objective.getTermVariables(subProblemIndex);
                final int currentSubProblemIndex = subProblemIndex;
                Vector variables = variableCopies.get(subProblemIndex);
                Vector multipliers = lagrangeMultipliers.get(subProblemIndex);
                Vector consensusVariables = Vectors.build(variableIndexes.length, currentPoint.type());
                consensusVariables.set(currentPoint.get(variableIndexes));
                if (currentIteration > 1 && subProblemSelectionMethod != SubProblemSelectionMethod.ALL)
                    for (int variableIndex : variableIndexes)
                        affectedConsensusVariables.add(variableIndex);
                if (solveSubProblem) {
                    subProblemTasks.add(Executors.callable(
                            () -> processSubProblem(currentSubProblemIndex,
                                                    variableIndexes,
                                                    variables,
                                                    consensusVariables,
                                                    multipliers,
                                                    variableCopiesSum)
                    ));
                    temporaryIndex++;
                }
                if (penaltyParameterSettingMethod == PenaltyParameterSettingMethod.ADAPTIVE
                        || checkForPrimalAndDualResidualConvergence
                        || subProblemSelectionMethod == SubProblemSelectionMethod.CONSENSUS_FOCUSED_SAMPLING) {
                    if (subProblemSelectionMethod != SubProblemSelectionMethod.ALL) {
                        primalToleranceAdder.add(-variables.norm(VectorNorm.L2_SQUARED));
                        dualToleranceAdder.add(-multipliers.norm(VectorNorm.L2_SQUARED));
                    }
                    residualComputationTasks.add(Executors.callable(
                            () -> computeResiduals(currentSubProblemIndex,
                                                   variables,
                                                   consensusVariables,
                                                   multipliers,
                                                   solveSubProblem)
                    ));
                }
            }
        }
        for (int constraintIndex = 0; constraintIndex < constraints.size(); constraintIndex++) {
            int[] variableIndexes = constraintsVariablesIndexes.get(constraintIndex);
            final int currentConstraintIndex = constraintIndex;
            int subProblemIndex = objective.getNumberOfTerms() + constraintIndex;
            Vector variables = variableCopies.get(subProblemIndex);
            Vector multipliers = lagrangeMultipliers.get(subProblemIndex);
            Vector consensusVariables = Vectors.build(variableIndexes.length, currentPoint.type());
            consensusVariables.set(currentPoint.get(variableIndexes));
            subProblemTasks.add(Executors.callable(
                    () -> processConstraint(currentConstraintIndex,
                                            variableIndexes,
                                            variables,
                                            consensusVariables,
                                            multipliers,
                                            variableCopiesSum)
            ));
            if (penaltyParameterSettingMethod == PenaltyParameterSettingMethod.ADAPTIVE
                    || checkForPrimalAndDualResidualConvergence
                    || subProblemSelectionMethod == SubProblemSelectionMethod.CONSENSUS_FOCUSED_SAMPLING) {
                if (subProblemSelectionMethod != SubProblemSelectionMethod.ALL) {
                    primalToleranceAdder.add(-variables.norm(VectorNorm.L2_SQUARED));
                    dualToleranceAdder.add(-multipliers.norm(VectorNorm.L2_SQUARED));
                }
                residualComputationTasks.add(Executors.callable(
                        () -> computeResiduals(subProblemIndex,
                                               variables,
                                               consensusVariables,
                                               multipliers,
                                               true)
                ));
            }
        }
        try {
            taskExecutor.invokeAll(subProblemTasks);
        } catch (InterruptedException e) {
            logger.error("Execution was interrupted while solving the sub-problems.");
        }
        if (currentIteration > 1 && subProblemSelectionMethod != SubProblemSelectionMethod.ALL) {
            int[] affectedConsensusVariablesIndexes =
                    ArrayUtils.toPrimitive(affectedConsensusVariables
                                                   .toArray(new Integer[affectedConsensusVariables.size()]));
            currentPoint.set(affectedConsensusVariablesIndexes,
                             variableCopiesSum.get(affectedConsensusVariablesIndexes)
                                     .divElementwise(variableCopiesCounts.get(affectedConsensusVariablesIndexes))
                                     .maxElementwiseInPlace(0)
                                     .minElementwiseInPlace(1));
        } else {
            currentPoint = variableCopiesSum
                    .divElementwise(variableCopiesCounts)
                    .maxElementwiseInPlace(0)
                    .minElementwiseInPlace(1);
        }
        if (penaltyParameterSettingMethod == PenaltyParameterSettingMethod.ADAPTIVE
                || subProblemSelectionMethod == SubProblemSelectionMethod.CONSENSUS_FOCUSED_SAMPLING
                || checkForPrimalAndDualResidualConvergence) {
            try {
                taskExecutor.invokeAll(residualComputationTasks);
            } catch (InterruptedException e) {
                logger.error("Execution was interrupted while computing the primal and dual residuals.");
            }
            if (penaltyParameterSettingMethod == PenaltyParameterSettingMethod.ADAPTIVE
                    || checkForPrimalAndDualResidualConvergence) {
                primalResidual = Math.sqrt(primalResidualSquaredTerms.sum());
                dualResidual = penaltyParameter *
                        Math.sqrt(currentPoint
                                          .sub(previousPoint)
                                          .map(x -> x * x)
                                          .multElementwise(variableCopiesCounts)
                                          .sum());
            }
        }
        penaltyParameterSettingMethod.updatePenaltyParameter(this); // Update the augmented Lagrangian penalty parameter
        if (checkForObjectiveConvergence || logObjectiveValue) {
            previousObjectiveValue = currentObjectiveValue;
            currentObjectiveValue = objective.getValue(currentPoint);
        }
        if (checkForGradientConvergence || logGradientNorm) {
            previousGradient = currentGradient;
            try {
                currentGradient = objective.getGradient(currentPoint);
            } catch (NonSmoothFunctionException e) {
                throw new UnsupportedOperationException(
                        "Trying to check for gradient convergence or log the gradient norm, " +
                                "while using a non-smooth objective function."
                );
            }
        }
    }

    private void processSubProblem(int subProblemIndex,
                                   int[] variableIndexes,
                                   Vector variables,
                                   Vector consensusVariables,
                                   Vector multipliers,
                                   Vector variableCopiesSum) {
        Vector temporaryVariables = variables.add(multipliers.div(penaltyParameter));
        synchronized (lock) {
            variableCopiesSum.set(variableIndexes,
                                  variableCopiesSum.get(variableIndexes)
                                          .sub(temporaryVariables));
        }
        multipliers.addInPlace(variables.sub(consensusVariables).mult(penaltyParameter));
        SubProblem subProblem = new SubProblem(subProblemIndex,
                                               variables,
                                               multipliers,
                                               consensusVariables,
                                               objective.getTerm(subProblemIndex),
                                               penaltyParameter);
        if (subProblemSolver != null)
            subProblemSolver.accept(subProblem);
        else
            solveSubProblem(subProblem);
        temporaryVariables = variables.add(multipliers.div(penaltyParameter));
        synchronized (lock) {
            variableCopiesSum.set(variableIndexes,
                                  variableCopiesSum.get(variableIndexes)
                                          .add(temporaryVariables));
        }
    }

    private void solveSubProblem(SubProblem subProblem) {
        subProblem.variables.set(
                new QuasiNewtonSolver.Builder(new SubProblemObjectiveFunction(subProblem.objectiveTerm,
                                                                              subProblem.consensusVariables,
                                                                              subProblem.multipliers,
                                                                              penaltyParameter),
                                              subProblem.variables).build().solve()
        );
    }

    private void processConstraint(int constraintIndex,
                                   int[] variableIndexes,
                                   Vector variables,
                                   Vector consensusVariables,
                                   Vector multipliers,
                                   Vector variableCopiesSum) {
        multipliers.addInPlace(variables.sub(consensusVariables).mult(penaltyParameter));
        try {
            variables.set(constraints.get(constraintIndex).project(consensusVariables));
        } catch (SingularMatrixException e) {
            logger.error("Singular matrix encountered in one of the problem constraints!");
        }
        Vector temporaryVariables = variables.add(multipliers.div(penaltyParameter));
        synchronized (lock) {
            variableCopiesSum.set(variableIndexes,
                                  variableCopiesSum.get(variableIndexes)
                                          .add(temporaryVariables));
        }
    }

    private void computeResiduals(int subProblemIndex,
                                  Vector variables,
                                  Vector consensusVariables,
                                  Vector multipliers,
                                  boolean variablesUpdated) {
        primalResidualSquaredTerms.set(subProblemIndex,
                                       variables.sub(consensusVariables).norm(VectorNorm.L2_SQUARED));
        if (variablesUpdated && checkForPrimalAndDualResidualConvergence) {
            primalToleranceAdder.add(variables.norm(VectorNorm.L2_SQUARED));
            dualToleranceAdder.add(multipliers.norm(VectorNorm.L2_SQUARED));
        }
    }

    public class SubProblem {
        public final int subProblemIndex;
        public final Vector variables;
        public final Vector multipliers;
        public final Vector consensusVariables;
        public final AbstractFunction objectiveTerm;
        public final double penaltyParameter;

        public SubProblem(int subProblemIndex,
                          Vector variables,
                          Vector multipliers,
                          Vector consensusVariables,
                          AbstractFunction objectiveTerm,
                          double penaltyParameter) {
            this.subProblemIndex = subProblemIndex;
            this.variables = variables;
            this.multipliers = multipliers;
            this.consensusVariables = consensusVariables;
            this.objectiveTerm = objectiveTerm;
            this.penaltyParameter = penaltyParameter;
        }
    }

    public static class SubProblemObjectiveFunction extends AbstractFunction {
        private final AbstractFunction subProblemObjectiveFunction;
        private final Vector consensusVariables;
        private final Vector lagrangeMultipliers;
        private final double penaltyParameter;

        public SubProblemObjectiveFunction(AbstractFunction subProblemObjectiveFunction,
                                           Vector consensusVariables,
                                           Vector lagrangeMultipliers,
                                           double penaltyParameter) {
            this.subProblemObjectiveFunction = subProblemObjectiveFunction;
            this.consensusVariables = consensusVariables;
            this.lagrangeMultipliers = lagrangeMultipliers;
            this.penaltyParameter = penaltyParameter;
        }

        @Override
        public boolean equals(Object other) {
            if (other == this) {
                return true;
            }

            if (!super.equals(other)) {
                return false;
            }

            if (!(other instanceof SubProblemObjectiveFunction)) {
                return false;
            }

            SubProblemObjectiveFunction rhs = (SubProblemObjectiveFunction)other;
            return new EqualsBuilder()
                    .append(this.consensusVariables, rhs.consensusVariables)
                    .append(this.lagrangeMultipliers, rhs.lagrangeMultipliers)
                    .append(this.penaltyParameter, rhs.penaltyParameter)
                    .append(this.subProblemObjectiveFunction, rhs.subProblemObjectiveFunction)
                    .isEquals();
        }

        @Override
        public int hashCode() {

            return new HashCodeBuilder(37, 73)
                    .append(super.hashCode())
                    .append(this.consensusVariables)
                    .append(this.lagrangeMultipliers)
                    .append(this.penaltyParameter)
                    .append(this.subProblemObjectiveFunction)
                    .toHashCode();

        }

        @Override
        protected double computeValue(Vector point) {
            return subProblemObjectiveFunction.getValue(point)
                    + penaltyParameter
                    * Math.pow(point.sub(consensusVariables)
                                       .add(lagrangeMultipliers.div(penaltyParameter))
                                       .norm(VectorNorm.L2_FAST), 2) / 2;
        }

        @Override
        protected Vector computeGradient(Vector point) throws NonSmoothFunctionException {
            return subProblemObjectiveFunction.getGradient(point)
                    .add(point.sub(consensusVariables)
                                 .add(lagrangeMultipliers.div(penaltyParameter))
                                 .mult(penaltyParameter));
        }

        @Override
        protected Matrix computeHessian(Vector point) throws NonSmoothFunctionException {
            return subProblemObjectiveFunction.getHessian(point)
                    .add(Matrix.generateIdentityMatrix(point.size()).multiply(penaltyParameter));
        }
    }

    public enum PenaltyParameterSettingMethod {
        CONSTANT {
            @Override
            public void updatePenaltyParameter(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver) {

            }
        },
        ADAPTIVE {
            @Override
            public void updatePenaltyParameter(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver) {
                if (solver.primalResidual > solver.mu * solver.dualResidual)
                    solver.penaltyParameter *= solver.tauIncrement;
                else if (solver.dualResidual > solver.mu * solver.primalResidual)
                    solver.penaltyParameter /= solver.tauDecrement;
            }
        };

        public abstract void updatePenaltyParameter(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver);
    }

    public enum SubProblemSelectionMethod {
        ALL {
            @Override
            public int[] selectSubProblems(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver) {
                int[] indexes = new int[solver.objective.getNumberOfTerms()];
                for (int index = 0; index < solver.objective.getNumberOfTerms(); index++)
                    indexes[index] = index;
                return indexes;
            }
        },
        UNIFORM_SAMPLING {
            @Override
            public int[] selectSubProblems(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver) {
                Integer[] indexes = new Integer[solver.objective.getNumberOfTerms()];
                for (int index = 0; index < solver.objective.getNumberOfTerms(); index++)
                    indexes[index] = index;
                return ArrayUtils.toPrimitive(
                        StatisticsUtilities.sampleWithoutReplacement(indexes, solver.numberOfSubProblemSamples)
                );
            }
        },
        CONSENSUS_FOCUSED_SAMPLING {
            @Override
            public int[] selectSubProblems(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver) {
                if (solver.currentIteration > 1) {
                    int[] indexes = new int[solver.objective.getNumberOfTerms()];
                    for (int index = 0; index < solver.objective.getNumberOfTerms(); index++)
                        indexes[index] = index;
                    return StatisticsUtilities
                            .sampleWithoutReplacement(indexes,
                                                      solver.primalResidualSquaredTerms.getDenseArray(),
                                                      solver.numberOfSubProblemSamples);
                } else {
                    return UNIFORM_SAMPLING.selectSubProblems(solver);
                }
            }
        };

        public abstract int[] selectSubProblems(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver);
    }
}
