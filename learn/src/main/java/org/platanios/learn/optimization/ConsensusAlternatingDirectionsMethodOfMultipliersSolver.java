package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.*;
import org.platanios.learn.optimization.constraint.AbstractConstraint;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.NonSmoothFunctionException;
import org.platanios.learn.optimization.function.SumFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * TODO: The current implementation is PSL-specific and not generic at all.
 * TODO: Only support p = 1 in PSL for now and only linear equality constraints.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class ConsensusAlternatingDirectionsMethodOfMultipliersSolver extends AbstractIterativeSolver {
    private final List<Vector> variableCopies = new ArrayList<>();
    private final List<Vector> lagrangeMultipliers = new ArrayList<>();

    private final Object lock = new Object();

    private final Vector variableCopiesCounts;
    private final double augmentedLagrangianParameter;
    private final ExecutorService taskExecutor;

    final SumFunction objective;
    final List<int[]> constraintsVariablesIndexes;
    final List<AbstractConstraint> constraints;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractIterativeSolver.AbstractBuilder<T> {
        protected final List<int[]> constraintsVariablesIndexes = new ArrayList<>();
        protected final List<AbstractConstraint> constraints = new ArrayList<>();

        protected double augmentedLagrangianParameter = 1;
        private int numberOfThreads = Runtime.getRuntime().availableProcessors();

        protected AbstractBuilder(SumFunction objective,
                                  Vector initialPoint) {
            super(objective, initialPoint);

            checkForPointConvergence = false;
            checkForObjectiveConvergence = false;
            checkForGradientConvergence = false;
        }

        public T addConstraint(int[] variableIndexes, AbstractConstraint constraint) {
            constraintsVariablesIndexes.add(variableIndexes);
            constraints.add(constraint);
            return self();
        }

        public T augmentedLagrangianParameter(double augmentedLagrangianParameter) {
            this.augmentedLagrangianParameter = augmentedLagrangianParameter;
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

    private ConsensusAlternatingDirectionsMethodOfMultipliersSolver(AbstractBuilder builder) {
        super(builder);
        objective = (SumFunction) builder.objective;
        constraintsVariablesIndexes = builder.constraintsVariablesIndexes;
        constraints = builder.constraints;
        augmentedLagrangianParameter = builder.augmentedLagrangianParameter;
        taskExecutor = Executors.newFixedThreadPool(builder.numberOfThreads);
        variableCopiesCounts = Vectors.dense(currentPoint.size());
        for (int[] variableIndexes : objective.getTermsVariables()) {
            Vector termPoint = Vectors.build(variableIndexes.length, currentPoint.type());
            termPoint.set(0, variableIndexes.length - 1, currentPoint.get(variableIndexes));
            variableCopies.add(termPoint);
            lagrangeMultipliers.add(Vectors.build(variableIndexes.length, currentPoint.type()));
            for (int variableIndex : variableIndexes)
                variableCopiesCounts.set(variableIndex, variableCopiesCounts.get(variableIndex) + 1);
        }
        for (int[] variableIndexes : constraintsVariablesIndexes) {
            Vector termPoint = Vectors.build(variableIndexes.length, currentPoint.type());
            termPoint.set(0, variableIndexes.length - 1, currentPoint.get(variableIndexes));
            variableCopies.add(termPoint);
            lagrangeMultipliers.add(Vectors.build(variableIndexes.length, currentPoint.type()));
            for (int variableIndex : variableIndexes)
                variableCopiesCounts.set(variableIndex, variableCopiesCounts.get(variableIndex) + 1);
        }
    }

    @Override
    public void performIterationUpdates() {
        previousPoint = currentPoint;
        previousGradient = currentGradient;
        previousObjectiveValue = currentObjectiveValue;
        Vector variableCopiesSum = Vectors.build(currentPoint.size(), currentPoint.type());
        List<Callable<Object>> subProblemTasks = new ArrayList<>();
        for (int subProblemIndex = 0; subProblemIndex < objective.getNumberOfTerms(); subProblemIndex++) {
            int[] variableIndexes = objective.getTermVariables(subProblemIndex);
            final int currentSubProblemIndex = subProblemIndex;
            subProblemTasks.add(Executors.callable(
                    () -> solveSubProblem(currentSubProblemIndex, variableIndexes, variableCopiesSum)
            ));
        }
        final int numberOfObjectiveTerms = objective.getNumberOfTerms();
        for (int constraintIndex = 0; constraintIndex < constraints.size(); constraintIndex++) {
            int[] variableIndexes = constraintsVariablesIndexes.get(constraintIndex);
            final int currentConstraintIndex = constraintIndex;
            subProblemTasks.add(Executors.callable(
                    () -> projectOnConstraint(currentConstraintIndex, numberOfObjectiveTerms, variableIndexes, variableCopiesSum)
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
//        if (checkForGradientConvergence)
        try {
            currentGradient = objective.getGradient(currentPoint);
        } catch (NonSmoothFunctionException ignored) { }
//        if (checkForObjectiveConvergence)
        currentObjectiveValue = objective.getValue(currentPoint);
    }

    private void solveSubProblem(int subProblemIndex, int[] variableIndexes, Vector variableCopiesSum) {
        Vector variables = variableCopies.get(subProblemIndex);
        Vector multipliers = lagrangeMultipliers.get(subProblemIndex);
        Vector consensusVariables = Vectors.build(variableIndexes.length, currentPoint.type());
        consensusVariables.set(0, variableIndexes.length - 1, currentPoint.get(variableIndexes));
        multipliers.addInPlace(variables
                                       .sub(consensusVariables)
                                       .mult(augmentedLagrangianParameter));
        variables.set(0, variables.size() - 1, consensusVariables.sub(multipliers.div(augmentedLagrangianParameter)));
        if (objective.getValue(variables, subProblemIndex) > 0) {
            variables.set(0, variables.size() - 1, new NewtonSolver.Builder(new SubProblemObjectiveFunction(objective.getTerm(subProblemIndex),
                                                                                                            consensusVariables,
                                                                                                            multipliers),
                                                                            variables).build().solve());
            if (objective.getValue(variables, subProblemIndex) < 0) {
                variables.set(0, variables.size() - 1, ((LinearFunction) objective.getTerm(subProblemIndex))
                        .projectToHyperplane(consensusVariables));
            }
        }
        Vector termPoint = Vectors.build(currentPoint.size(), currentPoint.type());
        termPoint.set(variableIndexes, variables.add(multipliers.div(augmentedLagrangianParameter)));
        synchronized (lock) {
            variableCopiesSum.addInPlace(termPoint);
        }
    }

    private void projectOnConstraint(int constraintIndex, int numberOfObjectiveTerms, int[] variableIndexes, Vector variableCopiesSum) {
        Vector variables = variableCopies.get(numberOfObjectiveTerms + constraintIndex);
        Vector multipliers = lagrangeMultipliers.get(numberOfObjectiveTerms + constraintIndex);
        Vector consensusVariables = Vectors.build(variableIndexes.length, currentPoint.type());
        consensusVariables.set(0, variableIndexes.length - 1, currentPoint.get(variableIndexes));
        multipliers.addInPlace(variables
                                       .sub(consensusVariables)
                                       .mult(augmentedLagrangianParameter));
        try {
            variables.set(0, variables.size() - 1, constraints.get(constraintIndex).project(consensusVariables));
        } catch (SingularMatrixException e) {
            logger.error("Singular matrix encountered in one of the problem constraints!");
        }
        Vector termPoint = Vectors.build(currentPoint.size(), currentPoint.type());
        termPoint.set(variableIndexes, variables.add(multipliers.div(augmentedLagrangianParameter)));
        synchronized (lock) {
            variableCopiesSum.addInPlace(termPoint);
        }
    }

    private class SubProblemObjectiveFunction extends AbstractFunction {
        private final AbstractFunction subProblemObjectiveFunction;
        private final Vector consensusVariables;
        private final Vector lagrangeMultipliers;

        SubProblemObjectiveFunction(AbstractFunction subProblemObjectiveFunction,
                                    Vector consensusVariables,
                                    Vector lagrangeMultipliers) {
            this.subProblemObjectiveFunction = subProblemObjectiveFunction;
            this.consensusVariables = consensusVariables;
            this.lagrangeMultipliers = lagrangeMultipliers;
        }

        @Override
        protected double computeValue(Vector point) {
            return subProblemObjectiveFunction.getValue(point)
                    + augmentedLagrangianParameter * Math.pow(point.sub(consensusVariables)
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
}
