package makina.learn.classification.active;

import makina.utilities.MathUtilities;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum SurpriseFunction {
    NEGATIVE_LOGARITHM {
        @Override
        public double surprise(double probability) {
            if (probability == 0)
                return -Math.log(epsilon);
            return -Math.log(probability);
        }
    },
    ONE_MINUS_PROBABILITY {
        @Override
        public double surprise(double probability) {
            return 1 - probability;
        }
    };

    private static final double epsilon = MathUtilities.computeMachineEpsilonDouble();

    public abstract double surprise(double probability);
}
