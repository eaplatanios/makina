package org.platanios.learn.classification;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.LabeledDataInstance;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.AbstractStochasticFunctionUsingDataSet;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.List;

/**
 * This class implements a binary logistic regression model that is trained using the stochastic gradient descent
 * algorithm.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractTrainableLogisticRegression
        extends LogisticRegressionPrediction implements TrainableClassifier<Vector, Integer> {
    /** The data used to train this model. */
    private DataSet<LabeledDataInstance<Vector, Integer>> trainingDataSet;
    /** Indicates whether /(L_1/) regularization is used. */
    protected boolean useL1Regularization;
    /** The /(L_1/) regularization weight used. This variable is only used when {@link #useL1Regularization} is set to
     * true. */
    protected double l1RegularizationWeight;
    /** Indicates whether /(L_2/) regularization is used. */
    protected boolean useL2Regularization;
    /** The /(L_2/) regularization weight used. This variable is only used when {@link #useL2Regularization} is set to
     * true. */
    protected double l2RegularizationWeight;
    protected int loggingLevel;

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
            extends LogisticRegressionPrediction.AbstractBuilder<T> {
        /** Indicates whether /(L_1/) regularization is used. */
        protected boolean useL1Regularization = false;
        /** The /(L_1/) regularization weight used. This variable is only used when {@link #useL1Regularization} is set
         * to true. */
        protected double l1RegularizationWeight = 1;
        /** Indicates whether /(L_2/) regularization is used. */
        protected boolean useL2Regularization = false;
        /** The /(L_2/) regularization weight used. This variable is only used when {@link #useL2Regularization} is set
         * to true. */
        protected double l2RegularizationWeight = 1;
        protected int loggingLevel = 0;

        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data. This constructor should be used if the logistic regression model that is being built is going
         * to be trained.
         *
         * @param   numberOfFeatures    The number of features used.
         */
        protected AbstractBuilder(int numberOfFeatures) {
            this.numberOfFeatures = numberOfFeatures;
        }

        protected AbstractBuilder(int numberOfFeatures, Vector weights) {
            super(numberOfFeatures, weights);
        }

        /**
         * Sets the {@link #useL1Regularization} field that indicates whether /(L_1/) regularization is used.
         *
         * @param   useL1Regularization The value to which to set the {@link #useL1Regularization} field.
         * @return                      This builder object itself. That is done so that we can use a nice and
         *                              expressive code format when we build objects using this builder class.
         */
        public T useL1Regularization(boolean useL1Regularization) {
            this.useL1Regularization = useL1Regularization;
            return self();
        }

        /**
         * Sets the {@link #l1RegularizationWeight} field that contains the value of the /(L_1/) regularization weight
         * used. This variable is only used when {@link #useL1Regularization} is set to true.
         *
         * @param   l1RegularizationWeight  The value to which to set the {@link #l1RegularizationWeight} field.
         * @return                          This builder object itself. That is done so that we can use a nice and
         *                                  expressive code format when we build objects using this builder class.
         */
        public T l1RegularizationWeight(double l1RegularizationWeight) {
            this.l1RegularizationWeight = l1RegularizationWeight;
            return self();
        }

        /**
         * Sets the {@link #useL2Regularization} field that indicates whether /(L_2/) regularization is used.
         *
         * @param   usel2Regularization The value to which to set the {@link #useL2Regularization} field.
         * @return                      This builder object itself. That is done so that we can use a nice and
         *                              expressive code format when we build objects using this builder class.
         */
        public T useL2Regularization(boolean usel2Regularization) {
            this.useL2Regularization = usel2Regularization;
            return self();
        }

        /**
         * Sets the {@link #l2RegularizationWeight} field that contains the value of the /(L_2/) regularization weight
         * used. This variable is only used when {@link #useL2Regularization} is set to true.
         *
         * @param   l2RegularizationWeight  The value to which to set the {@link #l2RegularizationWeight} field.
         * @return                          This builder object itself. That is done so that we can use a nice and
         *                                  expressive code format when we build objects using this builder class.
         */
        public T l2RegularizationWeight(double l2RegularizationWeight) {
            this.l2RegularizationWeight = l2RegularizationWeight;
            return self();
        }

        public T loggingLevel(int loggingLevel) {
            this.loggingLevel = loggingLevel;
            return self();
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
    protected AbstractTrainableLogisticRegression(AbstractBuilder<?> builder) {
        super(builder);

        useL1Regularization = builder.useL1Regularization;
        l1RegularizationWeight = builder.l1RegularizationWeight;
        useL2Regularization = builder.useL2Regularization;
        l2RegularizationWeight = builder.l2RegularizationWeight;
        loggingLevel = builder.loggingLevel;
    }

    @Override
    @SuppressWarnings("unchecked")
    public boolean train(DataSet<? extends LabeledDataInstance<Vector, Integer>> trainingDataSet) {
        this.trainingDataSet = (DataSet<LabeledDataInstance<Vector, Integer>>) trainingDataSet;
        try {
            train();
            return true;
        } catch(Exception e) {
            return false;
        }
    }

    /**
     * Trains this logistic regression model using the data provided while building this object.
     */
    protected abstract void train();

    /**
     * Class implementing the likelihood function for the binary logistic regression model. No function is provided to
     * compute the hessian matrix because no binary logistic regression class has to use it.
     */
    protected class LikelihoodFunction extends AbstractFunction {
        private Iterator<LabeledDataInstance<Vector, Integer>> trainingDataIterator;

        public LikelihoodFunction() {
            trainingDataIterator = trainingDataSet.iterator();
        }

        /**
         * Computes the value of the likelihood function for the binary logistic regression model.
         *
         * @param   weights The current weights vector.
         * @return          The value of the logistic regression likelihood function.
         */
        @Override
        public double computeValue(Vector weights) {
            double likelihood = 0;
            while (trainingDataIterator.hasNext()) {
                LabeledDataInstance<Vector, Integer> dataInstance = trainingDataIterator.next();
                double probability = weights.dot(dataInstance.features());
                likelihood += probability * (dataInstance.label() - 1) - Math.log(1 + Math.exp(-probability));
            }
            return -likelihood;
        }

        /**
         * Computes the gradient of the likelihood function for the binary logistic regression model.
         *
         * @param   weights The current weights vector.
         * @return          The gradient vector of the logistic regression likelihood function.
         */
        @Override
        public Vector computeGradient(Vector weights) {
            Vector gradient = Vectors.build(weights.size(), weights.type());
            while (trainingDataIterator.hasNext()) {
                LabeledDataInstance<Vector, Integer> dataInstance = trainingDataIterator.next();
                gradient.saxpyInPlace(
                        (1 / (1 + Math.exp(-weights.dot(dataInstance.features())))) - dataInstance.label(),
                        dataInstance.features()
                );
            }
            return gradient;
        }
    }

    /**
     * Class implementing the likelihood function for the binary logistic regression model for use with stochastic
     * solvers.
     */
    protected class StochasticLikelihoodFunction
            extends AbstractStochasticFunctionUsingDataSet<LabeledDataInstance<Vector, Integer>> {
        public StochasticLikelihoodFunction() {
            this.dataSet = trainingDataSet;
        }

        /**
         * Computes the gradient of the likelihood function for the multi-class logistic regression model.
         *
         * @param   weights     The current weights vector.
         * @param   dataBatch
         * @return              The gradient vector of the logistic regression likelihood function.
         */
        @Override
        public Vector estimateGradient(Vector weights, List<LabeledDataInstance<Vector, Integer>> dataBatch) {
            Vector gradient = Vectors.build(weights.size(), weights.type());
            for (LabeledDataInstance<Vector, Integer> dataInstance : dataBatch) {
                gradient.saxpyInPlace(
                        (1 / (1 + Math.exp(-weights.dot(dataInstance.features())))) - dataInstance.label(),
                        dataInstance.features()
                );
            }
            return gradient;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void write(OutputStream outputStream) throws IOException {
        super.write(outputStream);

        UnsafeSerializationUtilities.writeBoolean(outputStream, useL1Regularization);
        UnsafeSerializationUtilities.writeDouble(outputStream, l1RegularizationWeight);
        UnsafeSerializationUtilities.writeBoolean(outputStream, useL2Regularization);
        UnsafeSerializationUtilities.writeDouble(outputStream, l2RegularizationWeight);
        UnsafeSerializationUtilities.writeInt(outputStream, loggingLevel);
    }
}
