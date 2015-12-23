package org.platanios.learn.classification.active;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum SurpriseFunction {
    NEGATIVE_LOGARITHM {
        @Override
        public double surprise(double probability) {
            return -Math.log(probability);
        }
    },
    ONE_MINUS_PROBABILITY {
        @Override
        public double surprise(double probability) {
            return 1 - probability;
        }
    };

    public abstract double surprise(double probability);
}
