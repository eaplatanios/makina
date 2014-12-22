package org.platanios.learn.data;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum DataSetType {
    SIMPLE {
        @Override
        public Class getDataInstanceClass() {
            return DataInstance.class;
        }

        @Override
        public Class getDataInstanceWithoutFeaturesClass() {
            return DataInstanceBase.class;
        }
    },
    LABELED {
        @Override
        public Class getDataInstanceClass() {
            return LabeledDataInstance.class;
        }

        @Override
        public Class getDataInstanceWithoutFeaturesClass() {
            return LabeledDataInstanceBase.class;
        }
    },
    PREDICTED {
        @Override
        public Class getDataInstanceClass() {
            return PredictedDataInstance.class;
        }

        @Override
        public Class getDataInstanceWithoutFeaturesClass() {
            return PredictedDataInstanceBase.class;
        }
    };

    public abstract Class getDataInstanceClass();
    public abstract Class getDataInstanceWithoutFeaturesClass();
}
