package org.platanios.learn.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.optimization.constraint.AbstractConstraint;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.SumFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;

/**
 * TODO: The current implementation is PSL-specific and not generic at all.
 * TODO: Only support p = 1 in PSL for now and only linear equality constraints.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class ConsensusAlternatingDirectionsMethodOfMultipliersSolver implements Solver {
    private static final Logger logger = LogManager.getFormatterLogger("Optimization");
    private final List<Vector> variableCopies = new ArrayList<>();
    private final List<Vector> lagrangeMultipliers = new ArrayList<>();

    private final Vector variableCopiesCounts;
    private final double augmentedLagrangianParameter;
    private final int maximumNumberOfIterations;
    private final double pointChangeTolerance;
    private final double objectiveChangeTolerance;
    private final double gradientTolerance;
    private final boolean checkForPointConvergence;
    private final boolean checkForObjectiveConvergence;
    private final boolean checkForGradientConvergence;
    private final Function<Vector, Boolean> additionalCustomConvergenceCriterion;
    private final int loggingLevel;
    private final ExecutorService taskExecutor;

    private double pointChange;
    private double objectiveChange;
    private double gradientNorm;

    private boolean pointConverged = false;
    private boolean objectiveConverged = false;
    private boolean gradientConverged = false;

    final SumFunction objective;
    final List<int[]> constraintsVariablesIndexes;
    final List<AbstractConstraint> constraints;

    int currentIteration;
    Vector currentPoint;
    Vector previousPoint;
    Vector currentGradient;
    Vector previousGradient;
    double currentObjectiveValue;
    double previousObjectiveValue;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final SumFunction objective;
        protected final Vector initialPoint;

        protected final List<int[]> constraintsVariablesIndexes = new ArrayList<>();
        protected final List<AbstractConstraint> constraints = new ArrayList<>();

        protected double augmentedLagrangianParameter = 1;
        protected int maximumNumberOfIterations = 10000;
        protected double pointChangeTolerance = 1e-10;
        protected double objectiveChangeTolerance = 1e-10;
        protected double gradientTolerance = 1e-6;
        protected boolean checkForPointConvergence = true;
        protected boolean checkForObjectiveConvergence = true;
        protected boolean checkForGradientConvergence = true;
        private Function<Vector, Boolean> additionalCustomConvergenceCriterion = currentPoint -> false;
        private int loggingLevel = 0;
        private int numberOfThreads = Runtime.getRuntime().availableProcessors();

        protected AbstractBuilder(SumFunction objective,
                                  Vector initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
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

        public T maximumNumberOfIterations(int maximumNumberOfIterations) {
            this.maximumNumberOfIterations = maximumNumberOfIterations;
            return self();
        }

        public T pointChangeTolerance(double pointChangeTolerance) {
            this.pointChangeTolerance = pointChangeTolerance;
            return self();
        }

        public T objectiveChangeTolerance(double objectiveChangeTolerance) {
            this.objectiveChangeTolerance = objectiveChangeTolerance;
            return self();
        }

        public T gradientTolerance(double gradientTolerance) {
            this.gradientTolerance = gradientTolerance;
            return self();
        }

        public T checkForPointConvergence(boolean checkForPointConvergence) {
            this.checkForPointConvergence = checkForPointConvergence;
            return self();
        }

        public T checkForObjectiveConvergence(boolean checkForObjectiveConvergence) {
            this.checkForObjectiveConvergence = checkForObjectiveConvergence;
            return self();
        }

        public T checkForGradientConvergence(boolean checkForGradientConvergence) {
            this.checkForGradientConvergence = checkForGradientConvergence;
            return self();
        }

        public T additionalCustomConvergenceCriterion(Function<Vector, Boolean> additionalCustomConvergenceCriterion) {
            this.additionalCustomConvergenceCriterion = additionalCustomConvergenceCriterion;
            return self();
        }

        public T loggingLevel(int loggingLevel) {
            this.loggingLevel = loggingLevel;
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
        objective = builder.objective;
        constraintsVariablesIndexes = builder.constraintsVariablesIndexes;
        constraints = builder.constraints;
        augmentedLagrangianParameter = builder.augmentedLagrangianParameter;
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        pointChangeTolerance = builder.pointChangeTolerance;
        objectiveChangeTolerance = builder.objectiveChangeTolerance;
        gradientTolerance = builder.gradientTolerance;
        checkForPointConvergence = builder.checkForPointConvergence;
        checkForObjectiveConvergence = builder.checkForObjectiveConvergence;
        checkForGradientConvergence = builder.checkForGradientConvergence;
        additionalCustomConvergenceCriterion = builder.additionalCustomConvergenceCriterion;
        loggingLevel = builder.loggingLevel;
        taskExecutor = Executors.newFixedThreadPool(builder.numberOfThreads);
        currentPoint = builder.initialPoint;
        currentGradient = objective.getGradient(currentPoint);
        currentObjectiveValue = objective.getValue(currentPoint);
        currentIteration = 0;
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
    public Vector solve() {
        while (!checkTerminationConditions() && !additionalCustomConvergenceCriterion.apply(currentPoint)) {
            performIterationUpdates();
            currentIteration++;
            if ((loggingLevel == 1 && currentIteration % 1000 == 0)
                    || (loggingLevel == 2 && currentIteration % 100 == 0)
                    || (loggingLevel == 3 && currentIteration % 10 == 0)
                    || loggingLevel > 3)
                printIteration();
        }
        if (loggingLevel > 0)
            printTerminationMessage();
        return currentPoint;
    }

    private void performIterationUpdates() {
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
        for (int constraintIndex = 0; constraintIndex < constraints.size(); constraintIndex++) {
            int[] variableIndexes = constraintsVariablesIndexes.get(constraintIndex);
            final int currentConstraintIndex = constraintIndex;
            subProblemTasks.add(Executors.callable(
                    () -> projectOnConstraint(currentConstraintIndex, variableIndexes, variableCopiesSum)
            ));
        }
        try {
            taskExecutor.invokeAll(subProblemTasks);
        } catch (InterruptedException e) {
            logger.error("Execution was interrupted while solving the subproblems.");
        }
//        for (int subProblemIndex = 0; subProblemIndex < objective.getNumberOfTerms(); subProblemIndex++) {
//            int[] variableIndexes = objective.getTermVariables(subProblemIndex);
//            final int currentSubProblemIndex = subProblemIndex;
//            taskExecutor.submit(() -> solveSubProblem(currentSubProblemIndex, variableIndexes, variableCopiesSum));
//        }
//        for (int constraintIndex = 0; constraintIndex < constraints.size(); constraintIndex++) {
//            int[] variableIndexes = constraintsVariablesIndexes.get(constraintIndex);
//            final int currentConstraintIndex = constraintIndex;
//            taskExecutor.submit(() -> projectOnConstraint(currentConstraintIndex, variableIndexes, variableCopiesSum));
//        }
//        for (int subProblemIndex = 0; subProblemIndex < objective.getNumberOfTerms(); subProblemIndex++) {
//            int[] variableIndexes = objective.getTermVariables(subProblemIndex);
//            solveSubProblem(subProblemIndex, variableIndexes);
//            Vector termPoint = Vectors.build(currentPoint.size(), currentPoint.type());
//            termPoint.set(variableIndexes,
//                          variableCopies.get(subProblemIndex)
//                                  .add(lagrangeMultipliers.get(subProblemIndex).div(augmentedLagrangianParameter)));
//            variableCopiesSum.add(termPoint);
//        }
        currentPoint = variableCopiesSum.divElementwise(variableCopiesCounts);
        for (int variableIndex = 0; variableIndex < currentPoint.size(); variableIndex++)
            if (currentPoint.get(variableIndex) < 0)
                currentPoint.set(variableIndex, 0);
            else if (currentPoint.get(variableIndex) > 1)
                currentPoint.set(variableIndex, 1);
//        if (checkForGradientConvergence)
        currentGradient = objective.getGradient(currentPoint);
//        if (checkForObjectiveConvergence)
        currentObjectiveValue = objective.getValue(currentPoint);
    }

    private void solveSubProblem(int subProblemIndex, int[] variableIndexes, Vector variableCopiesSum) {
        Vector variables = variableCopies.get(subProblemIndex);
        Vector multipliers = lagrangeMultipliers.get(subProblemIndex);
        Vector consensusVariables = Vectors.build(variableIndexes.length, currentPoint.type());
        consensusVariables.set(0, variableIndexes.length - 1, currentPoint.get(variableIndexes));
        multipliers.addInPlace(variableCopies.get(subProblemIndex)
                                       .sub(consensusVariables)
                                       .mult(augmentedLagrangianParameter));
        variables.set(0, variables.size() - 1, consensusVariables.sub(multipliers.div(augmentedLagrangianParameter)));
        if (objective.getValue(variables, subProblemIndex) > 0) {
            variables = new NewtonSolver.Builder(new SubProblemObjectiveFunction(objective.getTerm(subProblemIndex),
                                                                                 consensusVariables,
                                                                                 multipliers),
                                                 variables).build().solve();
            if (objective.getValue(variables, subProblemIndex) < 0) {
                variables = ((LinearFunction) objective.getTerm(subProblemIndex))
                        .projectToHyperplane(consensusVariables);
            }
        }
        Vector termPoint = Vectors.build(currentPoint.size(), currentPoint.type());
        termPoint.set(variableIndexes, variables.add(multipliers.div(augmentedLagrangianParameter)));
        variableCopiesSum.add(termPoint);
    }

    private void projectOnConstraint(int constraintIndex, int[] variableIndexes, Vector variableCopiesSum) {
        Vector variables = variableCopies.get(constraintIndex);
        Vector multipliers = lagrangeMultipliers.get(constraintIndex);
        Vector consensusVariables = Vectors.build(variableIndexes.length, currentPoint.type());
        consensusVariables.set(0, variableIndexes.length - 1, currentPoint.get(variableIndexes));
        multipliers.addInPlace(variableCopies.get(constraintIndex)
                                       .sub(consensusVariables)
                                       .mult(augmentedLagrangianParameter));
        try {
            variables = constraints.get(constraintIndex).project(consensusVariables);
        } catch (NonSymmetricMatrixException e) {
            logger.error("Non-symmetric matrix encountered in one of the problem constraints!");
        } catch (NonPositiveDefiniteMatrixException e) {
            logger.error("Non-positive-definite matrix encountered in one of the problem constraints!");
        }
        Vector termPoint = Vectors.build(currentPoint.size(), currentPoint.type());
        termPoint.set(variableIndexes, variables.add(multipliers.div(augmentedLagrangianParameter)));
        variableCopiesSum.add(termPoint);
    }

    public boolean checkTerminationConditions() {
        if (currentIteration > 0) {
            if (currentIteration >= maximumNumberOfIterations)
                return true;
            if (checkForPointConvergence) {
                pointChange = currentPoint.sub(previousPoint).norm(VectorNorm.L2_FAST);
                pointConverged = pointChange <= pointChangeTolerance;
            }
            if (checkForObjectiveConvergence) {
                objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
                objectiveConverged = objectiveChange <= objectiveChangeTolerance;
            }
            if (checkForGradientConvergence) {
                gradientNorm = currentGradient.norm(VectorNorm.L2_FAST);
                gradientConverged = gradientNorm <= gradientTolerance;
            }
            return (checkForPointConvergence && pointConverged)
                    || (checkForObjectiveConvergence && objectiveConverged)
                    || (checkForGradientConvergence && gradientConverged);
        } else {
            return false;
        }
    }

    public void printIteration() {
        logger.info("Iteration #: %10d | Func. Eval. #: %10d | Objective Value: %20s " +
                            "| Objective Change: %20s | Point Change: %20s | Gradient Norm: %20s",
                    currentIteration,
                    objective.getNumberOfFunctionEvaluations(),
                    DECIMAL_FORMAT.format(currentObjectiveValue),
                    DECIMAL_FORMAT.format(objectiveChange),
                    DECIMAL_FORMAT.format(pointChange),
                    DECIMAL_FORMAT.format(gradientNorm));
    }

    public void printTerminationMessage() {
        if (pointConverged)
            logger.info("The L2 norm of the point change, %s, was below the convergence threshold of %s.",
                        DECIMAL_FORMAT.format(pointChange),
                        DECIMAL_FORMAT.format(pointChangeTolerance));
        if (objectiveConverged)
            logger.info("The relative change of the objective function value, %s, " +
                                "was below the convergence threshold of %s.",
                        DECIMAL_FORMAT.format(objectiveChange),
                        DECIMAL_FORMAT.format(objectiveChangeTolerance));
        if (gradientConverged)
            logger.info("The gradient norm became %s, which is less than the convergence threshold of %s.",
                        DECIMAL_FORMAT.format(gradientNorm),
                        DECIMAL_FORMAT.format(gradientTolerance));
        if (currentIteration >= maximumNumberOfIterations)
            logger.info("Reached the maximum number of allowed iterations, %d.", maximumNumberOfIterations);
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
        protected Vector computeGradient(Vector point) {
            return subProblemObjectiveFunction.getGradient(point)
                    .add(point.sub(consensusVariables)
                                 .add(lagrangeMultipliers.div(augmentedLagrangianParameter))
                                 .mult(augmentedLagrangianParameter));
        }

        @Override
        protected Matrix computeHessian(Vector point) {
            return subProblemObjectiveFunction.getHessian(point)
                    .add(Matrix.generateIdentityMatrix(point.size()).multiply(augmentedLagrangianParameter));
        }
    }
}
