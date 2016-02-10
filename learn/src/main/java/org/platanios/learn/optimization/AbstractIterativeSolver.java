package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorNorm;
import org.platanios.learn.optimization.function.AbstractFunction;

import java.util.function.Function;

/**
 * TODO: Add support for regularization.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractIterativeSolver implements Solver {
    final int maximumNumberOfIterations;
    final int maximumNumberOfFunctionEvaluations;
    final double pointChangeTolerance;
    final double objectiveChangeTolerance;
    final double gradientTolerance;
    final Function<Vector, Boolean> additionalCustomConvergenceCriterion;

    double pointChange;
    double objectiveChange;
    double gradientNorm;

    boolean pointConverged = false;
    boolean objectiveConverged = false;
    boolean gradientConverged = false;

    final boolean checkForPointConvergence;
    final boolean checkForObjectiveConvergence;
    final boolean checkForGradientConvergence;
    final int loggingLevel;
    final boolean logObjectiveValue;
    final boolean logGradientNorm;

    final AbstractFunction objective;

    int currentIteration;
    public Vector currentPoint;
    Vector previousPoint;
    Vector currentGradient;
    Vector previousGradient;
    double currentObjectiveValue;
    double previousObjectiveValue;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final AbstractFunction objective;

        protected Vector initialPoint;
        protected int maximumNumberOfIterations = 1000;
        protected int maximumNumberOfFunctionEvaluations = 1000000;
        protected double pointChangeTolerance = 1e-10;
        protected double objectiveChangeTolerance = 1e-10;
        protected double gradientTolerance = 1e-6;
        protected boolean checkForPointConvergence = true;
        protected boolean checkForObjectiveConvergence = true;
        protected boolean checkForGradientConvergence = true;
        protected Function<Vector, Boolean> additionalCustomConvergenceCriterion = currentPoint -> false;
        protected int loggingLevel = 0;
        
        private boolean logObjectiveValue = true;
        private boolean logGradientNorm = true;

        protected AbstractBuilder(AbstractFunction objective,
                                  Vector initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }

        public T initialPoint(Vector initialPoint) {
            this.initialPoint = initialPoint;
            return self();
        }

        public T maximumNumberOfIterations(int maximumNumberOfIterations) {
            this.maximumNumberOfIterations = maximumNumberOfIterations;
            return self();
        }

        public T maximumNumberOfFunctionEvaluations(int maximumNumberOfFunctionEvaluations) {
            this.maximumNumberOfFunctionEvaluations = maximumNumberOfFunctionEvaluations;
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

        public T logObjectiveValue(boolean logObjectiveValue) {
            this.logObjectiveValue = logObjectiveValue;
            return self();
        }

        public T logGradientNorm(boolean logGradientNorm) {
            this.logGradientNorm = logGradientNorm;
            return self();
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected AbstractIterativeSolver(AbstractBuilder<?> builder) {
        objective = builder.objective;
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        maximumNumberOfFunctionEvaluations = builder.maximumNumberOfFunctionEvaluations;
        pointChangeTolerance = builder.pointChangeTolerance;
        objectiveChangeTolerance = builder.objectiveChangeTolerance;
        gradientTolerance = builder.gradientTolerance;
        checkForPointConvergence = builder.checkForPointConvergence;
        checkForObjectiveConvergence = builder.checkForObjectiveConvergence;
        checkForGradientConvergence = builder.checkForGradientConvergence;
        additionalCustomConvergenceCriterion = builder.additionalCustomConvergenceCriterion;
        loggingLevel = builder.loggingLevel;
        logObjectiveValue = builder.logObjectiveValue;
        logGradientNorm = builder.logGradientNorm;
        currentPoint = builder.initialPoint;
        currentIteration = 0;
    }

    public int getCurrentIteration() {
        return currentIteration;
    }

    @Override
    public Vector solve() {
        if (loggingLevel > 0)
            logger.info("Optimization is starting.");
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

    public boolean checkTerminationConditions() {
        if (currentIteration > 0) {
            if (currentIteration >= maximumNumberOfIterations)
                return true;
            if (objective.getNumberOfFunctionEvaluations() >= maximumNumberOfFunctionEvaluations)
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
                if (this instanceof NonlinearConjugateGradientSolver) {
                    gradientNorm = Math.abs(currentGradient.max()) / (1 + Math.abs(currentObjectiveValue));
                    gradientConverged = gradientNorm <= gradientTolerance;
                } else {
                    gradientNorm = currentGradient.norm(VectorNorm.L2_FAST);
                    gradientConverged = gradientNorm <= gradientTolerance;
                }
            }
            return (checkForPointConvergence && pointConverged)
                    || (checkForObjectiveConvergence && objectiveConverged)
                    || (checkForGradientConvergence && gradientConverged);
        } else {
            return false;
        }
    }

    public void printIteration() {
        if (logObjectiveValue && logGradientNorm)
            logger.info("Iteration #: %10d | Func. Eval. #: %10d | Objective Value: %20s " +
                                "| Objective Change: %20s | Point Change: %20s | Gradient Norm: %20s",
                        currentIteration,
                        objective.getNumberOfFunctionEvaluations(),
                        DECIMAL_FORMAT.format(currentObjectiveValue),
                        DECIMAL_FORMAT.format(objectiveChange),
                        DECIMAL_FORMAT.format(pointChange),
                        DECIMAL_FORMAT.format(gradientNorm));
        else if (logObjectiveValue)
            logger.info("Iteration #: %10d | Func. Eval. #: %10d | Objective Value: %20s " +
                                "| Objective Change: %20s | Point Change: %20s",
                        currentIteration,
                        objective.getNumberOfFunctionEvaluations(),
                        DECIMAL_FORMAT.format(currentObjectiveValue),
                        DECIMAL_FORMAT.format(objectiveChange),
                        DECIMAL_FORMAT.format(pointChange));
        else if (logGradientNorm)
            logger.info("Iteration #: %10d | Func. Eval. #: %10d | Point Change: %20s | Gradient Norm: %20s",
                        currentIteration,
                        objective.getNumberOfFunctionEvaluations(),
                        DECIMAL_FORMAT.format(pointChange),
                        DECIMAL_FORMAT.format(gradientNorm));
        else
            logger.info("Iteration #: %10d | Func. Eval. #: %10d | Point Change: %20s",
                        currentIteration,
                        objective.getNumberOfFunctionEvaluations(),
                        DECIMAL_FORMAT.format(pointChange));
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
        if (objective.getNumberOfFunctionEvaluations() >= maximumNumberOfFunctionEvaluations)
            logger.info("Reached the maximum number of allowed objective function evaluations, %d.",
                        maximumNumberOfFunctionEvaluations);
    }

    public abstract void performIterationUpdates();
}
