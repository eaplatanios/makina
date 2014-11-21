package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.StochasticGradientDescentSolver;
import org.platanios.learn.optimization.StochasticSolverStepSize;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * This class implements a binary logistic regression model that is trained using the stochastic gradient descent
 * algorithm.
 *
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegressionSGD extends AbstractTrainableLogisticRegression {
    private static final long serialVersionUID = -6803380467546567631L;

    private boolean sampleWithReplacement;
    private int maximumNumberOfIterations;
    private int maximumNumberOfIterationsWithNoPointChange;
    private double pointChangeTolerance;
    private boolean checkForPointConvergence;
    private int batchSize;
    private StochasticSolverStepSize stepSize;
    private double[] stepSizeParameters;

    /**
     * This abstract class needs to be extended by the builder of its parent binary logistic regression class. It
     * provides an implementation for those parts of those builders that are common. This is basically part of a small
     * "hack" so that we can have inheritable builder classes.
     *
     * @param   <T> This type corresponds to the type of the final object to be built. That is, the super class of the
     *              builder class that extends this class, which in this case will be the
     *              {@link LogisticRegressionSGD} class.
     */
    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractTrainableLogisticRegression.AbstractBuilder<T> {
        protected boolean sampleWithReplacement = false;
        protected int maximumNumberOfIterations = 10000;
        protected int maximumNumberOfIterationsWithNoPointChange = 5;
        protected double pointChangeTolerance = 1e-10;
        protected boolean checkForPointConvergence = true;
        protected int batchSize = 100;
        protected StochasticSolverStepSize stepSize = StochasticSolverStepSize.SCALED;
        protected double[] stepSizeParameters = new double[] { 10, 0.75 };

        protected AbstractBuilder(int numberOfFeatures) {
            super(numberOfFeatures);
        }

        protected AbstractBuilder(int numberOfFeatures, Vector weights) {
            super(numberOfFeatures, weights);
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

        @Override
        public LogisticRegressionSGD build() {
            return new LogisticRegressionSGD(this);
        }
    }

    /**
     * The builder class for this class. This is basically part of a small "hack" so that we can have inheritable
     * builder classes.
     */
    public static class Builder extends AbstractBuilder<Builder> {
        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data using the stochastic gradient descent algorithm.
         *
         * @param   numberOfFeatures    The number of features used.
         */
        public Builder(int numberOfFeatures) {
            super(numberOfFeatures);
        }

        public Builder(int numberOfFeatures, Vector weights) {
            super(numberOfFeatures, weights);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }

    /**
     * Constructs a binary logistic regression object that uses the stochastic gradient descent algorithm to train the
     * model, given an appropriate builder object. This constructor can only be used from within the builder class of
     * this class.
     *
     * @param   builder The builder object to use.
     */
    private LogisticRegressionSGD(AbstractBuilder<?> builder) {
        super(builder);

        sampleWithReplacement = builder.sampleWithReplacement;
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        maximumNumberOfIterationsWithNoPointChange = builder.maximumNumberOfIterationsWithNoPointChange;
        pointChangeTolerance = builder.pointChangeTolerance;
        checkForPointConvergence = builder.checkForPointConvergence;
        batchSize = builder.batchSize;
        stepSize = builder.stepSize;
        stepSizeParameters = builder.stepSizeParameters;
    }

    @Override
    public ClassifierType type() {
        return ClassifierType.LOGISTIC_REGRESSION_SGD;
    }

    /** {@inheritDoc} */
    @Override
    protected void train() {
        weights = new StochasticGradientDescentSolver.Builder(new StochasticLikelihoodFunction(), weights)
                .sampleWithReplacement(sampleWithReplacement)
                .maximumNumberOfIterations(maximumNumberOfIterations)
                .maximumNumberOfIterationsWithNoPointChange(maximumNumberOfIterationsWithNoPointChange)
                .pointChangeTolerance(pointChangeTolerance)
                .checkForPointConvergence(checkForPointConvergence)
                .batchSize(batchSize)
                .stepSize(stepSize)
                .stepSizeParameters(stepSizeParameters)
                .useL1Regularization(useL1Regularization)
                .l1RegularizationWeight(l1RegularizationWeight)
                .useL2Regularization(useL2Regularization)
                .l2RegularizationWeight(l2RegularizationWeight)
                .loggingLevel(loggingLevel)
                .build().solve();
    }

    /** {@inheritDoc} */
    @Override
    public void writeObject(ObjectOutputStream outputStream) throws IOException {
        super.writeObject(outputStream);

        outputStream.writeBoolean(sampleWithReplacement);
        outputStream.writeInt(maximumNumberOfIterations);
        outputStream.writeInt(maximumNumberOfIterationsWithNoPointChange);
        outputStream.writeDouble(pointChangeTolerance);
        outputStream.writeBoolean(checkForPointConvergence);
        outputStream.writeInt(batchSize);
        outputStream.writeObject(stepSize);
        outputStream.writeObject(stepSizeParameters);
    }

    /** {@inheritDoc} */
    @Override
    public void readObject(ObjectInputStream inputStream) throws IOException, ClassNotFoundException {
        super.readObject(inputStream);

        sampleWithReplacement = inputStream.readBoolean();
        maximumNumberOfIterations = inputStream.readInt();
        maximumNumberOfIterationsWithNoPointChange = inputStream.readInt();
        pointChangeTolerance = inputStream.readDouble();
        checkForPointConvergence = inputStream.readBoolean();
        batchSize = inputStream.readInt();
        stepSize = (StochasticSolverStepSize) inputStream.readObject();
        stepSizeParameters = (double[]) inputStream.readObject();
    }
}
