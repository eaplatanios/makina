package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.*;
import org.platanios.learn.optimization.constraint.AbstractConstraint;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.NonSmoothFunctionException;
import org.platanios.learn.optimization.function.SumFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * TODO: The current implementation is PSL-specific and not generic at all.
 * TODO: Only support p = 1 in PSL for now and only linear equality constraints.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class ConsensusAlternatingDirectionsMethodOfMultipliersSolver extends AbstractIterativeSolver {
    private final Object lock = new Object();
    private final double primalResidualTolerance;
    private final double dualResidualTolerance;
    private final boolean checkForPrimalResidualConvergence;
    private final boolean checkForDualResidualConvergence;
    private final boolean checkForPrimalAndDualResidualConvergence;
    private final List<Vector> variableCopies = new ArrayList<>();
    private final List<Vector> lagrangeMultipliers = new ArrayList<>();

    private final Vector variableCopiesCounts;
    private final PenaltyParameterSettingMethod penaltyParameterSettingMethod;
    private final double mu;
    private final double tauIncrement;
    private final double tauDecrement;
    private final Consumer<SubProblem> subProblemSolver;
    private final ExecutorService taskExecutor;

    private final SumFunction objective;
    private final List<int[]> constraintsVariablesIndexes;
    private final List<AbstractConstraint> constraints;

    private double penaltyParameter;

    private double primalResidualSquared;
    private double dualResidualSquared;

    private boolean primalResidualConverged = false;
    private boolean dualResidualConverged = false;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractIterativeSolver.AbstractBuilder<T> {
        protected final List<int[]> constraintsVariablesIndexes = new ArrayList<>();
        protected final List<AbstractConstraint> constraints = new ArrayList<>();

        protected PenaltyParameterSettingMethod penaltyParameterSettingMethod = PenaltyParameterSettingMethod.ADAPTIVE;
        protected double mu = 10;
        protected double tauIncrement = 2;
        protected double tauDecrement = 2;
        protected double penaltyParameter = 1e-4;
        protected Consumer<SubProblem> subProblemSolver = null;
        protected int numberOfThreads = Runtime.getRuntime().availableProcessors();

        protected double primalResidualTolerance = 1e-5;
        protected double dualResidualTolerance = 1e-5;
        protected boolean checkForPrimalResidualConvergence = false;
        protected boolean checkForDualResidualConvergence = false;
        protected boolean checkForPrimalAndDualResidualConvergence = true;

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

        public T penaltyParameterSettingMethod(PenaltyParameterSettingMethod penaltyParameterSettingMethod) {
            this.penaltyParameterSettingMethod = penaltyParameterSettingMethod;
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

        public T primalResidualTolerance(double primalResidualTolerance) {
            this.primalResidualTolerance = primalResidualTolerance;
            return self();
        }

        public T dualResidualTolerance(double dualResidualTolerance) {
            this.dualResidualTolerance = dualResidualTolerance;
            return self();
        }

        public T checkForPrimalResidualConvergence(boolean checkForPrimalResidualConvergence) {
            this.checkForPrimalResidualConvergence = checkForPrimalResidualConvergence;
            return self();
        }

        public T checkForDualResidualConvergence(boolean checkForDualResidualConvergence) {
            this.checkForDualResidualConvergence = checkForDualResidualConvergence;
            return self();
        }

        public T checkForPrimalAndDualResidualConvergence(boolean checkForPrimalAndDualResidualConvergence) {
            this.checkForPrimalAndDualResidualConvergence = checkForPrimalAndDualResidualConvergence;
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
        primalResidualTolerance = builder.primalResidualTolerance;
        dualResidualTolerance = builder.dualResidualTolerance;
        checkForPrimalResidualConvergence = builder.checkForPrimalResidualConvergence;
        checkForDualResidualConvergence = builder.checkForDualResidualConvergence;
        checkForPrimalAndDualResidualConvergence = builder.checkForPrimalAndDualResidualConvergence;
        penaltyParameterSettingMethod = builder.penaltyParameterSettingMethod;
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
        if (super.checkTerminationConditions())
            return true;
        if (currentIteration > 0) {
            if (checkForPrimalResidualConvergence || checkForPrimalAndDualResidualConvergence) {
                if (penaltyParameterSettingMethod != PenaltyParameterSettingMethod.ADAPTIVE) {
                    primalResidualSquared = 0;
                    for (int subProblemIndex = 0; subProblemIndex < objective.getNumberOfTerms(); subProblemIndex++) {
                        primalResidualSquared += variableCopies.get(subProblemIndex)
                                .sub(currentPoint.get(objective.getTermVariables(subProblemIndex)))
                                .norm(VectorNorm.L2_SQUARED);
                    }
                    for (int constraintIndex = 0; constraintIndex < constraints.size(); constraintIndex++) {
                        primalResidualSquared += variableCopies.get(objective.getNumberOfTerms() + constraintIndex)
                                .sub(currentPoint.get(constraintsVariablesIndexes.get(constraintIndex)))
                                .norm(VectorNorm.L2_SQUARED);
                    }
                }
                primalResidualConverged = primalResidualSquared <= primalResidualTolerance;
            }
            if (checkForDualResidualConvergence || checkForPrimalAndDualResidualConvergence) {
                if (penaltyParameterSettingMethod != PenaltyParameterSettingMethod.ADAPTIVE)
                    dualResidualSquared = objective.getNumberOfTerms()
                            * Math.pow(penaltyParameter, 2)
                            * currentPoint.sub(previousPoint).norm(VectorNorm.L2_SQUARED);
                dualResidualConverged = dualResidualSquared <= dualResidualTolerance;
            }
            return (checkForPrimalResidualConvergence && primalResidualConverged)
                    || (checkForDualResidualConvergence && dualResidualConverged)
                    || (checkForPrimalAndDualResidualConvergence && primalResidualConverged && dualResidualConverged);
        } else {
            return false;
        }
    }

    @Override
    public void printTerminationMessage() {
        super.printTerminationMessage();
        if (primalResidualConverged)
            logger.info("The L2 norm of the primal residual, %s, was below the convergence threshold of %s.",
                        DECIMAL_FORMAT.format(primalResidualSquared),
                        DECIMAL_FORMAT.format(primalResidualTolerance));
        if (dualResidualConverged)
            logger.info("The L2 norm of the dual residual, %s, was below the convergence threshold of %s.",
                        DECIMAL_FORMAT.format(dualResidualSquared),
                        DECIMAL_FORMAT.format(dualResidualTolerance));
    }

    @Override
    public void performIterationUpdates() {
        previousPoint = currentPoint;
        Vector variableCopiesSum = Vectors.build(currentPoint.size(), currentPoint.type());
        List<Callable<Object>> subProblemTasks = new ArrayList<>();
        for (int subProblemIndex = 0; subProblemIndex < objective.getNumberOfTerms(); subProblemIndex++) {
            int[] variableIndexes = objective.getTermVariables(subProblemIndex);
            final int currentSubProblemIndex = subProblemIndex;
            subProblemTasks.add(Executors.callable(
                    () -> processSubProblem(currentSubProblemIndex, variableIndexes, variableCopiesSum)
            ));
        }
        final int numberOfObjectiveTerms = objective.getNumberOfTerms();
        for (int constraintIndex = 0; constraintIndex < constraints.size(); constraintIndex++) {
            int[] variableIndexes = constraintsVariablesIndexes.get(constraintIndex);
            final int currentConstraintIndex = constraintIndex;
            subProblemTasks.add(Executors.callable(
                    () -> processConstraint(currentConstraintIndex,
                                            variableIndexes,
                                            variableCopiesSum,
                                            numberOfObjectiveTerms)
            ));
        }
        try {
            taskExecutor.invokeAll(subProblemTasks);
        } catch (InterruptedException e) {
            logger.error("Execution was interrupted while solving the subproblems.");
        }
        currentPoint = variableCopiesSum.divElementwise(variableCopiesCounts);
        for (int variableIndex = 0; variableIndex < currentPoint.size(); variableIndex++)
            if (currentPoint.get(variableIndex) < 0)
                currentPoint.set(variableIndex, 0);
            else if (currentPoint.get(variableIndex) > 1)
                currentPoint.set(variableIndex, 1);
        if (penaltyParameterSettingMethod == PenaltyParameterSettingMethod.ADAPTIVE) {
            primalResidualSquared = 0;
            for (int subProblemIndex = 0; subProblemIndex < objective.getNumberOfTerms(); subProblemIndex++) {
                primalResidualSquared += variableCopies.get(subProblemIndex)
                        .sub(currentPoint.get(objective.getTermVariables(subProblemIndex)))
                        .norm(VectorNorm.L2_SQUARED);
            }
            for (int constraintIndex = 0; constraintIndex < constraints.size(); constraintIndex++) {
                primalResidualSquared += variableCopies.get(objective.getNumberOfTerms() + constraintIndex)
                        .sub(currentPoint.get(constraintsVariablesIndexes.get(constraintIndex)))
                        .norm(VectorNorm.L2_SQUARED);
            }
            dualResidualSquared = objective.getNumberOfTerms()
                    * penaltyParameter * penaltyParameter
                    * currentPoint.sub(previousPoint).norm(VectorNorm.L2_SQUARED);
            penaltyParameterSettingMethod.updatePenaltyParameter(this);
        }
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

    private void processSubProblem(int subProblemIndex, int[] variableIndexes, Vector variableCopiesSum) {
        Vector variables = variableCopies.get(subProblemIndex);
        Vector multipliers = lagrangeMultipliers.get(subProblemIndex);
        Vector consensusVariables = Vectors.build(variableIndexes.length, currentPoint.type());
        consensusVariables.set(currentPoint.get(variableIndexes));
        multipliers.addInPlace(variables.sub(consensusVariables).mult(penaltyParameter));
        variables.set(consensusVariables.sub(multipliers.div(penaltyParameter)));
        SubProblem subProblem = new SubProblem(variables,
                                               multipliers,
                                               consensusVariables,
                                               objective.getTerm(subProblemIndex),
                                               penaltyParameter);
        if (subProblemSolver != null)
            subProblemSolver.accept(subProblem);
        else
            solveSubProblem(subProblem);
        Vector termPoint = Vectors.build(currentPoint.size(), currentPoint.type());
        termPoint.set(variableIndexes, variables.add(multipliers.div(penaltyParameter)));
        synchronized (lock) {
            variableCopiesSum.addInPlace(termPoint);
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
                                   Vector variableCopiesSum,
                                   int numberOfObjectiveTerms) {
        Vector variables = variableCopies.get(numberOfObjectiveTerms + constraintIndex);
        Vector multipliers = lagrangeMultipliers.get(numberOfObjectiveTerms + constraintIndex);
        Vector consensusVariables = Vectors.build(variableIndexes.length, currentPoint.type());
        consensusVariables.set(currentPoint.get(variableIndexes));
        multipliers.addInPlace(variables.sub(consensusVariables).mult(penaltyParameter));
        try {
            variables.set(constraints.get(constraintIndex).project(consensusVariables));
        } catch (SingularMatrixException e) {
            logger.error("Singular matrix encountered in one of the problem constraints!");
        }
        Vector termPoint = Vectors.build(currentPoint.size(), currentPoint.type());
        termPoint.set(variableIndexes, variables.add(multipliers.div(penaltyParameter)));
        synchronized (lock) {
            variableCopiesSum.addInPlace(termPoint);
        }
    }

    public class SubProblem {
        public final Vector variables;
        public final Vector multipliers;
        public final Vector consensusVariables;
        public final AbstractFunction objectiveTerm;
        public final double augmentedLagrangianParameter;

        public SubProblem(Vector variables,
                          Vector multipliers,
                          Vector consensusVariables,
                          AbstractFunction objectiveTerm,
                          double augmentedLagrangianParameter) {
            this.variables = variables;
            this.multipliers = multipliers;
            this.consensusVariables = consensusVariables;
            this.objectiveTerm = objectiveTerm;
            this.augmentedLagrangianParameter = augmentedLagrangianParameter;
        }
    }

    public static class SubProblemObjectiveFunction extends AbstractFunction {
        private final AbstractFunction subProblemObjectiveFunction;
        private final Vector consensusVariables;
        private final Vector lagrangeMultipliers;
        private final double augmentedLagrangianParameter;

        public SubProblemObjectiveFunction(AbstractFunction subProblemObjectiveFunction,
                                           Vector consensusVariables,
                                           Vector lagrangeMultipliers,
                                           double augmentedLagrangianParameter) {
            this.subProblemObjectiveFunction = subProblemObjectiveFunction;
            this.consensusVariables = consensusVariables;
            this.lagrangeMultipliers = lagrangeMultipliers;
            this.augmentedLagrangianParameter = augmentedLagrangianParameter;
        }

        @Override
        protected double computeValue(Vector point) {
            return subProblemObjectiveFunction.getValue(point)
                    + augmentedLagrangianParameter
                    * Math.pow(point.sub(consensusVariables)
                                       .add(lagrangeMultipliers.div(augmentedLagrangianParameter))
                                       .norm(VectorNorm.L2_FAST), 2) / 2;
        }

        @Override
        protected Vector computeGradient(Vector point) throws NonSmoothFunctionException {
            return subProblemObjectiveFunction.getGradient(point)
                    .add(point.sub(consensusVariables)
                                 .add(lagrangeMultipliers.div(augmentedLagrangianParameter))
                                 .mult(augmentedLagrangianParameter));
        }

        @Override
        protected Matrix computeHessian(Vector point) throws NonSmoothFunctionException {
            return subProblemObjectiveFunction.getHessian(point)
                    .add(Matrix.generateIdentityMatrix(point.size()).multiply(augmentedLagrangianParameter));
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
                if (solver.primalResidualSquared > solver.mu * solver.mu * solver.dualResidualSquared)
                    solver.penaltyParameter *= solver.tauIncrement;
                else if (solver.dualResidualSquared > solver.mu * solver.mu * solver.primalResidualSquared)
                    solver.penaltyParameter /= solver.tauDecrement;
            }
        };

        public abstract void updatePenaltyParameter(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver);
    }
}
