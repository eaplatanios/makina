package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * @author Emmanouil Antonios Platanios.
 */
public enum ClassifierType {
    LOGISTIC_REGRESSION_PREDICTION {
        @Override
        public Classifier<Vector, Integer> build(ObjectInputStream inputStream) throws IOException {
            return new LogisticRegressionPrediction.Builder(inputStream).build();
        }
    },
    LOGISTIC_REGRESSION_SGD {
        @Override
        public TrainableClassifier<Vector, Integer> build(ObjectInputStream inputStream) throws IOException {
            return new LogisticRegressionAdaGrad.Builder(inputStream).build();
        }
    },
    LOGISTIC_REGRESSION_ADAGRAD {
        @Override
        public TrainableClassifier<Vector, Integer> build(ObjectInputStream inputStream) throws IOException {
            return new LogisticRegressionAdaGrad.Builder(inputStream).build();
        }
    };

    public abstract Classifier build(ObjectInputStream inputStream) throws IOException;
}
