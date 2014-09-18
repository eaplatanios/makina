package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.DenseVector;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractStochasticIterativeSolver implements Solver {
    private final int maximumNumberOfIterations;
    private final int maximumNumberOfIterationsWithNoPointChange;
    private final double pointChangeTolerance;
    private final boolean checkForPointConvergence;
    private final int batchSize;
    private final StochasticSolverStepSize stepSize;
    private final double[] stepSizeParameters;

    private double pointChange;
    private int numberOfIterationsWithNoPointChange = 0;
    private boolean pointConverged = false;

    final AbstractStochasticFunction objective;

    int currentIteration;
    Vector currentPoint;
    Vector previousPoint;
    Vector currentGradient;
    Vector currentDirection;
    double currentStepSize;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final AbstractStochasticFunction objective;
        protected final double[] initialPoint;

        protected boolean sampleWithReplacement = true;
        protected int maximumNumberOfIterations = 10000;
        protected int maximumNumberOfIterationsWithNoPointChange = 1;
        protected double pointChangeTolerance = 1e-10;
        protected boolean checkForPointConvergence = true;
        protected int batchSize = 100;
        protected StochasticSolverStepSize stepSize = StochasticSolverStepSize.SCALED;
        protected double[] stepSizeParameters = new double[] { 10, 0.75 };

        protected AbstractBuilder(AbstractStochasticFunction objective,
                                  double[] initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }

        public T sampleWithReplacement(boolean sampleWithReplacement) {
            this.sampleWithReplacement = sampleWithReplacement;
            return self();
        }

        public T maximumNumberOfIterations(int maximumNumberOfIterations) {
            this.maximumNumberOfIterations = maximumNumberOfIterations;
            return self();
        }

        public T maximumNumberOfIterationsWithNoPointChange(int maximumNumberOfIterationsWithNoPointChange) {
            this.maximumNumberOfIterationsWithNoPointChange = maximumNumberOfIterationsWithNoPointChange;
            return self();
        }

        public T pointChangeTolerance(double pointChangeTolerance) {
            this.pointChangeTolerance = pointChangeTolerance;
            return self();
        }

        public T checkForPointConvergence(boolean checkForPointConvergence) {
            this.checkForPointConvergence = checkForPointConvergence;
            return self();
        }

        public T batchSize(int batchSize) {
            this.batchSize = batchSize;
            return self();
        }

        public T stepSize(StochasticSolverStepSize stepSize) {
            this.stepSize = stepSize;
            return self();
        }

        public T stepSizeParameters(double... stepSizeParameters) {
            this.stepSizeParameters = stepSizeParameters;
            return self();
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractStochasticFunction objective,
                       double[] initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected AbstractStochasticIterativeSolver(AbstractBuilder<?> builder) {
        objective = builder.objective;
        objective.setSampleWithReplacement(builder.sampleWithReplacement);
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        maximumNumberOfIterationsWithNoPointChange = builder.maximumNumberOfIterationsWithNoPointChange;
        pointChangeTolerance = builder.pointChangeTolerance;
        checkForPointConvergence = builder.checkForPointConvergence;
        batchSize = builder.batchSize;
        stepSize = builder.stepSize;
        stepSizeParameters = builder.stepSizeParameters;
        currentPoint = new DenseVector(builder.initialPoint);
        currentGradient = objective.getGradientEstimate(currentPoint, batchSize);
        currentIteration = 0;
    }

    @Override
    public Vector solve() {
        printHeader();
        while (!checkTerminationConditions()) {
            performIterationUpdates();
            currentIteration++;
            printIteration();
        }
        printTerminationMessage();
        return currentPoint;
    }

    public void updateStepSize() {
        currentStepSize = stepSize.compute(currentIteration, stepSizeParameters);
    }

    public void performIterationUpdates() {
        updateDirection();
        updateStepSize();
        previousPoint = currentPoint;
        updatePoint();
        currentGradient = objective.getGradientEstimate(currentPoint, batchSize);
    }

    public boolean checkTerminationConditions() {
        if (currentIteration > 0) {
            if (currentIteration >= maximumNumberOfIterations) {
                return true;
            }

            if (checkForPointConvergence) {
                pointChange = currentPoint.subtract(previousPoint).computeL2Norm();
                numberOfIterationsWithNoPointChange =
                        (pointChange <= pointChangeTolerance) ? numberOfIterationsWithNoPointChange + 1 : 0;
                if (numberOfIterationsWithNoPointChange >= maximumNumberOfIterationsWithNoPointChange) {
                    pointConverged = true;
                }
            }

            return checkForPointConvergence && pointConverged;
        } else {
            return false;
        }
    }

    public void printHeader() {
        System.out.println("|----------------" +
                                   "----------------------|");
        System.out.format("| %13s | %20s |%n",
                          "Iteration #",
                          "Point Change");
        System.out.println("|===============|" +
                                   "======================|");
    }

    public void printIteration() {
        System.out.format("| %13d | %20s |%n",
                          currentIteration,
                          DECIMAL_FORMAT.format(pointChange));
    }

    public void printTerminationMessage() {
        System.out.println("|----------------" +
                                   "----------------------|\n");

        if (pointConverged) {
            System.out.println("The L2 norm of the point change, "
                                       + DECIMAL_FORMAT.format(pointChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(pointChangeTolerance)
                                       + " for more than "
                                       + maximumNumberOfIterationsWithNoPointChange
                                       + " iterations!");
        }
        if (currentIteration >= maximumNumberOfIterations) {
            System.out.println("Reached the maximum number of allowed iterations ("
                                       + maximumNumberOfIterations
                                       + ")!");
        }
    }

    public abstract void updateDirection();
    public abstract void updatePoint();
}
