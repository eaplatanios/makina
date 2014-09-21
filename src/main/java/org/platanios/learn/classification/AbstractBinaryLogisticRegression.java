package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Utilities;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorFactory;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;

import java.util.Arrays;
import java.util.List;

/**
 * This abstract class provides some functionality that is common to all binary logistic regression classes. All those
 * classes should extend this class.
 * TODO: Add bias term.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractBinaryLogisticRegression {
    /** The data used to train this model. */
    private final DataInstance<Vector, Integer>[] trainingData;
    /** The number of features used. */
    protected final int numberOfFeatures;

    /** The weights (i.e., parameters) used by this logistic regression model. */
    protected Vector weights;

    /**
     * This abstract class needs to be extended by the builders of all binary logistic regression classes. It provides
     * an implementation for those parts of those builders that are common. This is basically part of a small "hack" so
     * that we can have inheritable builder classes.
     *
     * @param   <T> This type corresponds to the type of the final object to be built. That is, the super class of the
     *              builder class that extends this class.
     */
    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        /** A self-reference to this builder class. This is basically part of a small "hack" so that we can have
         * inheritable builder classes. */
        protected abstract T self();

        /** The data used to train the logistic regression model to be built. */
        private final DataInstance<Vector, Integer>[] trainingData;
        /** The number of features used. */
        private final int numberOfFeatures;

        /** Indicates whether sparse vectors should be used. */
        private boolean sparse = false;

        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data.
         *
         * @param   trainingData    The training data with which the binary logistic regression model to be built by
         *                          this builder will be trained.
         */
        protected AbstractBuilder(DataInstance<Vector, Integer>[] trainingData) {
            this.trainingData = trainingData;
            numberOfFeatures = trainingData[0].getFeatures().size();
        }

        /**
         * Sets the {@link #sparse} field that indicates whether sparse vectors should be used.
         *
         * @param   sparse  The value to which to set the {@link #sparse} field.
         * @return          This builder object itself. That is done so that we can use a nice and expressive code
         *                  format when we build objects using this builder class.
         */
        public T sparse(boolean sparse) {
            this.sparse = sparse;
            return self();
        }
    }

    /**
     * The builder class for this abstract class. This is basically part of a small "hack" so that we can have
     * inheritable builder classes.
     */
    public static class Builder extends AbstractBuilder<Builder> {
        /**
         * Constructs a builder object for a binary logistic regression model that will be trained with the provided
         * training data.
         *
         * @param   trainingData    The training data with which the binary logistic regression model to be built by
         *                          this builder will be trained.
         */
        public Builder(DataInstance<Vector, Integer>[] trainingData) {
            super(trainingData);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }

    /**
     * Constructs a binary logistic regression object given an appropriate builder object. This constructor can only be
     * used from within the builder class of this class.
     *
     * @param   builder The builder object to use.
     */
    protected AbstractBinaryLogisticRegression(AbstractBuilder<?> builder) {
        trainingData = builder.trainingData;
        numberOfFeatures = builder.numberOfFeatures;
        if (builder.sparse) {
            weights = VectorFactory.build(numberOfFeatures, VectorType.SPARSE);
        } else {
            weights = VectorFactory.build(numberOfFeatures, VectorType.DENSE);
        }
    }

    /**
     * Trains this logistic regression model using the data provided while building this object.
     */
    public abstract void train();

    /**
     * Predict the probability of the class label being 1 for some data instance.
     *
     * @param   dataInstance    The data instance for which the probability is computed.
     * @return                  The probability of the class label being 1 for the given data instance.
     */
    public double predict(DataInstance<Vector, Integer> dataInstance) {
        double probability = weights.dot(dataInstance.getFeatures());
        probability = Math.exp(probability - Utilities.computeLogSumExp(0, probability));
        return probability;
    }

    /**
     * Predict the probabilities of the class labels being equal to 1 for a set of data instances. One probability value
     * is provided for each data instance in the set.
     *
     * @param   dataInstances   The set of data instances for which the probabilities are computed in the form of an
     *                          array.
     * @return                  The probabilities of the class labels being equal to 1 for the given set of data
     *                          instances.
     */
    public double[] predict(DataInstance<Vector, Integer>[] dataInstances) {
        double[] probabilities = new double[dataInstances.length];
        for (int i = 0; i < dataInstances.length; i++) {
            probabilities[i] = predict(dataInstances[i]);
            double probability = weights.dot(dataInstances[i].getFeatures());
            probabilities[i] = Math.exp(probability - Utilities.computeLogSumExp(0, probability));
        }
        return probabilities;
    }

    /**
     * Class implementing the likelihood function for the binary logistic regression model. No function is provided to
     * compute the hessian matrix because no binary logistic regression class has to use it.
     */
    protected class LikelihoodFunction extends AbstractFunction {
        /**
         * Computes the value of the likelihood function for the binary logistic regression model.
         *
         * @param   weights The current weights vector.
         * @return          The value of the logistic regression likelihood function.
         */
        @Override
        public double computeValue(Vector weights) {
            double likelihood = 0;
            for (DataInstance<Vector, Integer> dataInstance : trainingData) {
                double probability = weights.dot(dataInstance.getFeatures());
                likelihood += probability * dataInstance.getLabel() - Utilities.computeLogSumExp(0, probability);
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
            Vector gradient = VectorFactory.build(weights.size(), weights.type());
            for (DataInstance<Vector, Integer> dataInstance : trainingData) {
                double probability = weights.dot(dataInstance.getFeatures());
                probability = Math.exp(probability - Utilities.computeLogSumExp(0, probability));
                gradient.addInPlace(dataInstance.getFeatures().mult(probability - dataInstance.getLabel()));
            }
            return gradient;
        }
    }

    /**
     * Class implementing the likelihood function for the binary logistic regression model for use with stochastic
     * solvers.
     */
    protected class StochasticLikelihoodFunction extends AbstractStochasticFunction<DataInstance<Vector, Integer>> {
        public StochasticLikelihoodFunction() {
            // Using the method Arrays.asList so that the training data array is not duplicated. The newly created list
            // is backed by the existing array and any changes made to the list also "write through" to the array.
            this.data = Arrays.asList(trainingData);
        }

        /**
         * Computes the gradient of the likelihood function for the multi-class logistic regression model.
         *
         * @param weights The current weights vector.
         * @return The gradient vector of the logistic regression likelihood function.
         */
        @Override
        public Vector estimateGradient(Vector weights, List<DataInstance<Vector, Integer>> dataBatch) {
            Vector gradient = VectorFactory.build(weights.size(), weights.type());
            for (DataInstance<Vector, Integer> dataInstance : dataBatch) {
                double probability = weights.dot(dataInstance.getFeatures());
                probability = Math.exp(probability - Utilities.computeLogSumExp(0, probability));
                gradient.addInPlace(dataInstance.getFeatures().mult(probability - dataInstance.getLabel()));
            }
            return gradient;
        }
    }
}
