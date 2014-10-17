package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.AdaptiveGradientSolver;
import org.platanios.learn.optimization.StochasticSolverStepSize;

import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * This class implements a binary logistic regression model that is trained using the adaptive gradient algorithm.
 *
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegressionAdaGrad extends AbstractTrainableLogisticRegression {
    /** The adaptive gradient solver that is used to train this binary logistic regression model. */
    private final AdaptiveGradientSolver solver;

    /**
     * This abstract class needs to be extended by the builder of its parent binary logistic regression class. It
     * provides an implementation for those parts of those builders that are common. This is basically part of a small
     * "hack" so that we can have inheritable builder classes.
     *
     * @param   <T> This type corresponds to the type of the final object to be built. That is, the super class of the
     *              builder class that extends this class, which in this case will be the
     *              {@link LogisticRegressionAdaGrad} class.
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

        public AbstractBuilder(DataInstance<Vector, Integer>[] trainingData) {
            super(trainingData);
        }

        protected AbstractBuilder(DataInstance<Vector, Integer>[] trainingData, ObjectInputStream inputStream)
                throws IOException {
            super(trainingData, inputStream);
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
        public LogisticRegressionAdaGrad build() {
            return new LogisticRegressionAdaGrad(this);
        }
    }

    /**
     * The builder class for this class. This is basically part of a small "hack" so that we can have inheritable
     * builder classes.
     */
    public static class Builder extends AbstractBuilder<Builder> {
        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data using the adaptive gradient algorithm.
         *
         * @param   trainingData    The training data with which the binary logistic regression model to be built by
         *                          this builder will be trained.
         */
        public Builder(DataInstance<Vector, Integer>[] trainingData) {
            super(trainingData);
        }

        /**
         * Constructs a builder object for a binary logistic regression model and loads the model parameters (i.e., the
         * weight vectors from the provided input stream. This constructor should be used if the logistic regression
         * model that is being built is going to be used for making predictions alone (i.e., no training is supported).
         *
         * @param   inputStream The input stream from which to read the model parameters from.
         * @throws java.io.IOException
         */
        public Builder(DataInstance<Vector, Integer>[] trainingData, ObjectInputStream inputStream) throws IOException {
            super(trainingData, inputStream);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }

    /**
     * Constructs a binary logistic regression object that uses the adaptive gradient algorithm to train the model,
     * given an appropriate builder object. This constructor can only be used from within the builder class of this
     * class.
     *
     * @param   builder The builder object to use.
     */
    private LogisticRegressionAdaGrad(AbstractBuilder<?> builder) {
        super(builder);

        solver = new AdaptiveGradientSolver.Builder(new StochasticLikelihoodFunction(), weights)
                .sampleWithReplacement(builder.sampleWithReplacement)
                .maximumNumberOfIterations(builder.maximumNumberOfIterations)
                .maximumNumberOfIterationsWithNoPointChange(builder.maximumNumberOfIterationsWithNoPointChange)
                .pointChangeTolerance(builder.pointChangeTolerance)
                .checkForPointConvergence(builder.checkForPointConvergence)
                .batchSize(builder.batchSize)
                .stepSize(builder.stepSize)
                .stepSizeParameters(builder.stepSizeParameters)
                .useL1Regularization(builder.useL1Regularization)
                .l1RegularizationWeight(builder.l1RegularizationWeight)
                .useL2Regularization(builder.useL2Regularization)
                .l2RegularizationWeight(builder.l2RegularizationWeight)
                .loggingLevel(builder.loggingLevel)
                .build();
    }

    /** {@inheritDoc} */
    @Override
    public void train() {
        weights = solver.solve();
    }
}
