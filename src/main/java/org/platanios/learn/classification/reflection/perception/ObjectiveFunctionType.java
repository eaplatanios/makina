package org.platanios.learn.classification.reflection.perception;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * An enumeration of all possible objective function types that are currently supported by our implementation.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum ObjectiveFunctionType {
    /** An objective function that quantifies the dependencies between the error rates of the  different functions. The
     * actual implementation of this objective function types is {@link DependencyObjectiveFunction}. */
    DEPENDENCY {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            return new DependencyObjectiveFunction(errorRates, hessianIndexKeyMapping);
        }
    },
    SCALED_DEPENDENCY {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new NotImplementedException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_ACROSS_DOMAINS {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new NotImplementedException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_ACROSS_DOMAINS {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new NotImplementedException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    L2_NORM {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new NotImplementedException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_AND_L2_NORM {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new NotImplementedException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_AND_L2_NORM {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new NotImplementedException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_ACROSS_DOMAINS_AND_L2_NORM {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new NotImplementedException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_ACROSS_DOMAINS_AND_L2_NORM {
        @Override
        public ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new NotImplementedException();
        }
    }; // TODO: Implement this objective function and add all relevant documentation.

    public abstract ObjectiveFunction build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping);
}
