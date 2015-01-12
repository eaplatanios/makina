package org.platanios.learn.classification;

import java.io.IOException;
import java.io.InputStream;
import java.util.Set;
import java.util.TreeSet;

/**
 * @author Emmanouil Antonios Platanios.
 */
public enum ClassifierType {
    LOGISTIC_REGRESSION_PREDICTION {
        @Override
        public Set<ClassifierType> getStorageCompatibleTypes() {
            Set<ClassifierType> storageCompatibleTypesSet = new TreeSet<>();
            storageCompatibleTypesSet.add(LOGISTIC_REGRESSION_PREDICTION);
            storageCompatibleTypesSet.add(LOGISTIC_REGRESSION_SGD);
            storageCompatibleTypesSet.add(LOGISTIC_REGRESSION_ADAGRAD);
            return storageCompatibleTypesSet;
        }

        @Override
        public LogisticRegressionPrediction read(InputStream inputStream) throws IOException {
            return LogisticRegressionPrediction.read(inputStream);
        }
    },
    LOGISTIC_REGRESSION_SGD {
        @Override
        public Set<ClassifierType> getStorageCompatibleTypes() {
            Set<ClassifierType> storageCompatibleTypesSet = new TreeSet<>();
            storageCompatibleTypesSet.add(LOGISTIC_REGRESSION_SGD);
            return storageCompatibleTypesSet;
        }

        @Override
        public LogisticRegressionSGD read(InputStream inputStream) throws IOException {
            return LogisticRegressionSGD.read(inputStream);
        }
    },
    LOGISTIC_REGRESSION_ADAGRAD {
        @Override
        public Set<ClassifierType> getStorageCompatibleTypes() {
            Set<ClassifierType> storageCompatibleTypesSet = new TreeSet<>();
            storageCompatibleTypesSet.add(LOGISTIC_REGRESSION_ADAGRAD);
            return storageCompatibleTypesSet;
        }

        @Override
        public LogisticRegressionAdaGrad read(InputStream inputStream) throws IOException {
            return LogisticRegressionAdaGrad.read(inputStream);
        }
    };

    public abstract Set<ClassifierType> getStorageCompatibleTypes();
    public abstract Classifier read(InputStream inputStream) throws IOException;
}
