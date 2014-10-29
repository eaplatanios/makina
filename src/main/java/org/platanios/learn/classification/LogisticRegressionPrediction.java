package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.math.matrix.VectorType;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

/**
 * This abstract class provides some functionality that is common to all binary logistic regression classes. All those
 * classes should extend this class.
 * TODO: Add bias term.
 *
 * @author Emmanouil Antonios Platanios
 */
public class LogisticRegressionPrediction implements Classifier<Vector, Integer> {
    /** The number of features used. */
    protected final int numberOfFeatures;
    /** Indicates whether sparse vectors are being used or not. */
    protected final boolean sparse;

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

        /** The number of features used. */
        protected int numberOfFeatures;
        /** Indicates whether sparse vectors should be used or not. */
        private boolean sparse = false;
        /** The weights (i.e., parameters) used by this logistic regression model. */
        protected Vector weights = null;

        protected AbstractBuilder() { }

        /**
         * Constructs a builder object for a binary logistic regression model and loads the model parameters (i.e., the
         * weight vectors from the provided input stream. This constructor should be used if the logistic regression
         * model that is being built is going to be used for making predictions alone (i.e., no training is supported).
         *
         * @param   inputStream The input stream from which to read the model parameters from.
         * @throws  IOException
         */
        protected AbstractBuilder(ObjectInputStream inputStream) throws IOException {
            numberOfFeatures = inputStream.readInt();
            sparse = inputStream.readBoolean();
            if (sparse) {
                weights = Vectors.build(inputStream, VectorType.SPARSE);
            } else {
                weights = Vectors.build(inputStream, VectorType.DENSE);
            }
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

        public LogisticRegressionPrediction build() {
            return new LogisticRegressionPrediction(this);
        }
    }

    /**
     * The builder class for this abstract class. This is basically part of a small "hack" so that we can have
     * inheritable builder classes.
     */
    public static class Builder extends AbstractBuilder<Builder> {
        /**
         * Constructs a builder object for a binary logistic regression model and loads the model parameters (i.e., the
         * weight vectors from the provided input stream. This constructor should be used if the logistic regression
         * model that is being built is going to be used for making predictions alone (i.e., no training is supported).
         *
         * @param   inputStream The input stream from which to read the model parameters from.
         * @throws  IOException
         */
        public Builder(ObjectInputStream inputStream) throws IOException {
            super(inputStream);
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
    protected LogisticRegressionPrediction(AbstractBuilder<?> builder) {
        numberOfFeatures = builder.numberOfFeatures;
        sparse = builder.sparse;
        if (builder.weights != null) {
            weights = builder.weights;
        } else {
            if (sparse) {
                weights = Vectors.build(numberOfFeatures, VectorType.SPARSE);
            } else {
                weights = Vectors.build(numberOfFeatures, VectorType.DENSE);
            }
        }
    }

    @Override
    public ClassifierType type() {
        return ClassifierType.LOGISTIC_REGRESSION_PREDICTION;
    }

    /**
     * Predict the probability of the class label being 1 for some data instance.
     *
     * @param   dataInstance    The data instance for which the probability is computed.
     * @return                  The probability of the class label being 1 for the given data instance.
     */
    public double predict(DataInstance<Vector, Integer> dataInstance) {
        return 1 / (1 + Math.exp(-weights.dot(dataInstance.features())));
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
    public double[] predict(List<DataInstance<Vector, Integer>> dataInstances) {
        double[] probabilities = new double[dataInstances.size()];
        for (int i = 0; i < dataInstances.size(); i++) {
            probabilities[i] = 1 / (1 + Math.exp(-weights.dot(dataInstances.get(i).features())));
        }
        return probabilities;
    }

    public List<DataInstance<Vector, Integer>> predictInPlace(List<DataInstance<Vector, Integer>> dataInstances) {
        for (int i = 0; i < dataInstances.size(); i++) {
            DataInstance<Vector, Integer> currentDataInstance = dataInstances.get(i);
            double probability = 1 / (1 + Math.exp(-weights.dot(currentDataInstance.features())));
            if (probability >= 0.5) {
                dataInstances.set(i,
                                  new DataInstance.Builder<>(currentDataInstance)
                                          .label(1)
                                          .probability(probability)
                                          .build());
            } else {
                dataInstances.set(i,
                                  new DataInstance.Builder<>(currentDataInstance)
                                          .label(0)
                                          .probability(1 - probability)
                                          .build());
            }
        }
        return dataInstances;
    }

    /**
     * Writes the current logistic regression model to the provided output stream.
     *
     * @param   outputStream    The output stream to write the current logistic regression model to.
     * @throws  IOException
     */
    @Override
    public void writeModelToStream(ObjectOutputStream outputStream) throws IOException {
        outputStream.writeInt(numberOfFeatures);
        outputStream.writeBoolean(sparse);
        weights.writeToStream(outputStream);
    }
}
