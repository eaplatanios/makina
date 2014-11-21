package org.platanios.learn.classification;

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
            storageCompatibleTypesSet.add(LOGISTIC_REGRESSION_SGD);
            storageCompatibleTypesSet.add(LOGISTIC_REGRESSION_ADAGRAD);
            return storageCompatibleTypesSet;
        }
    },
    LOGISTIC_REGRESSION_SGD {
        @Override
        public Set<ClassifierType> getStorageCompatibleTypes() {
            return new TreeSet<>();
        }
    },
    LOGISTIC_REGRESSION_ADAGRAD {
        @Override
        public Set<ClassifierType> getStorageCompatibleTypes() {
            return new TreeSet<>();
        }
    };

    public abstract Set<ClassifierType> getStorageCompatibleTypes();
}
