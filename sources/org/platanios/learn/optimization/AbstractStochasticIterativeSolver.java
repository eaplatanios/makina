package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractStochasticIterativeSolver implements Solver {
    private final int maximumNumberOfIterations;
    private final double pointChangeTolerance;
    private final boolean checkForPointConvergence;
    private final int batchSize;
    private final double tau;
    private final double kappa;

    private double pointChange;
    private boolean pointConverged = false;

    final AbstractStochasticFunction objective;

    int currentIteration;
    Vector currentPoint;
    Vector previousPoint;
    Vector currentGradient;
    Vector currentDirection;
    double currentStepSize;

    public static abstract class Builder<T extends AbstractStochasticIterativeSolver> {
        protected final AbstractStochasticFunction objective;
        protected final double[] initialPoint;

        protected int maximumNumberOfIterations = 10000;
        protected double pointChangeTolerance = 1e-10;
        protected boolean checkForPointConvergence = true;
        protected int batchSize = 100;
        protected double tau = 10;
        protected double kappa = 0.75;

        protected Builder(AbstractStochasticFunction objective,
                          double[] initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }

        public Builder<T> maximumNumberOfIterations(int maximumNumberOfIterations) {
            this.maximumNumberOfIterations = maximumNumberOfIterations;
            return this;
        }

        public Builder<T> pointChangeTolerance(double pointChangeTolerance) {
            this.pointChangeTolerance = pointChangeTolerance;
            return this;
        }

        public Builder<T> checkForPointConvergence(boolean checkForPointConvergence) {
            this.checkForPointConvergence = checkForPointConvergence;
            return this;
        }

        public Builder<T> batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder<T> tau(double tau) {
            if (tau < 0) {
                throw new IllegalArgumentException("The value of the tau parameter must be >= 0.");
            }
            this.tau = tau;
            return this;
        }

        public Builder<T> kappa(double kappa) {
            if (kappa <= 0.5 || kappa > 1) {
                throw new IllegalArgumentException("The value of the kappa parameter must be in the interval (0.5,1].");
            }
            this.kappa = kappa;
            return this;
        }

        public abstract T build();
    }

    AbstractStochasticIterativeSolver(Builder builder) {
        objective = builder.objective;
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        pointChangeTolerance = builder.pointChangeTolerance;
        checkForPointConvergence = builder.checkForPointConvergence;
        batchSize = builder.batchSize;
        tau = builder.tau;
        kappa = builder.kappa;

        currentPoint = new Vector(builder.initialPoint);
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
        currentStepSize = Math.pow(tau + currentIteration + 1, -kappa);
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
                pointConverged = pointChange <= pointChangeTolerance;
            }

            return checkForPointConvergence && pointConverged;
        } else {
            return false;
        }
    }

    public void printHeader() {
        System.out.println("|----------------" +
                                   "----------------" +
                                   "-----------------------" +
                                   "-----------------------" +
                                   "-----------------------" +
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
                                   "----------------" +
                                   "-----------------------" +
                                   "-----------------------" +
                                   "-----------------------" +
                                   "----------------------|\n");

        if (pointConverged) {
            System.out.println("The L2 norm of the point change, "
                                       + DECIMAL_FORMAT.format(pointChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(pointChangeTolerance)
                                       + "!");
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
