package org.platanios.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.primitives.Ints;

import java.util.List;

/**
 * An enumeration of all possible objective function types that are currently supported by our implementation. Each type
 * also includes the implementation of the corresponding objective function.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum AgreementIntegratorObjective {
    /** An objective function that quantifies the dependencies between the error rates of the  different functions. */
    DEPENDENCY {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            return new DependencyObjectiveFunction(errorRates, hessianIndexKeyMapping);
        }

        /**
         * An implementation of an objective function that quantifies the dependencies between the error rates of the
         * different functions. The goal of the optimization procedure using this objective function is to minimize it,
         * subject to some equality and some inequality constraints, and obtain an estimate for the error rates.
         */
        class DependencyObjectiveFunction implements Function {
            /** The error rates structure used for all calculations (this structure contains the power set indexing
             * information used to index the error rates over all possible power sets of functions as a simple
             * one-dimensional array). */
            private final ErrorRatesPowerSetVector errorRates;
            /** A mapping used for specifying the sparse format of the Lagrangian Hessian matrix. More specifically,
             * each entry of the two-dimensional array is equal to the corresponding index of that Hessian entry in the
             * sparse, one-dimensional, representation of the Hessian matrix. */
            private final int[][] hessianIndexKeyMapping;

            /**
             * Initializes all parameters needed for computing the objective function computeValue, its derivatives with
             * respect to its input variables, and its Hessian with respect to the input variables.
             *
             * @param   errorRates              The error rates structure used for all calculations (this structure
             *                                  contains the power set indexing information used to index the error
             *                                  rates over all possible power sets of functions as a simple
             *                                  one-dimensional array).
             * @param   hessianIndexKeyMapping  A mapping used for specifying the sparse format of the Lagrangian
             *                                  Hessian matrix. More specifically, each entry of the two-dimensional
             *                                  array is equal to the corresponding index of that Hessian entry in the
             *                                  sparse, one-dimensional, representation of the Hessian matrix.
             */
            public DependencyObjectiveFunction(ErrorRatesPowerSetVector errorRates,
                                               int[][] hessianIndexKeyMapping) {
                this.errorRates = errorRates;
                this.hessianIndexKeyMapping = hessianIndexKeyMapping;
            }

            /** {@inheritDoc} */
            @Override
            public void computeValue(double[] point, double[] objectiveValue) {
                objectiveValue[0] = 0;
                for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
                    List<Integer> entryKey = entry.getKey();
                    int k = entryKey.size();
                    if (k > 1) {
                        double tempProduct = 1;
                        for (int index : entryKey)
                            tempProduct *= point[index];
                        objectiveValue[0] += Math.pow(point[entry.getValue()] - tempProduct, 2);
                    }
                }
            }

            /** {@inheritDoc} */
            @Override
            public void computeGradient(double[] point, double[] objectiveGradient) {
                for (int i = 0; i < objectiveGradient.length; i++)
                    objectiveGradient[i] = 0;
                for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
                    if (entry.getKey().size() > 1) {
                        double tempProduct = 1;
                        List<Integer> indexes = entry.getKey();
                        for (int index : indexes)
                            tempProduct *= point[index];
                        double term = point[entry.getValue()] - tempProduct;
                        objectiveGradient[entry.getValue()] += 2 * term;
                        for (int index : indexes)
                            objectiveGradient[index] -= 2 * term * tempProduct / point[index];
                    }
                }
            }

            /** {@inheritDoc} */
            @Override
            public void computeHessian(double[] point, double[] objectiveHessian) {
                for (int i = 0; i < objectiveHessian.length; i++)
                    objectiveHessian[i] = 0;
                for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
                    int[] entryKey = Ints.toArray(entry.getKey());
                    if (entryKey.length > 1) {
                        double tempProduct = 1;
                        for (int index : entryKey)
                            tempProduct *= point[index];
                        int jointTermIndex = entry.getValue();
                        objectiveHessian[hessianIndexKeyMapping[jointTermIndex][jointTermIndex]] = 2;
                        for (int i = 0; i < entryKey.length; i++) {
                            objectiveHessian[hessianIndexKeyMapping[entryKey[i]][jointTermIndex]] =
                                    2 * tempProduct / point[entryKey[i]];
                            objectiveHessian[hessianIndexKeyMapping[entryKey[i]][entryKey[i]]] +=
                                    2 * point[jointTermIndex] * tempProduct
                                            / Math.pow(point[entryKey[i]], 2);
                            // Note that the (map) keys of indexKeyMapping are in ascending order by construction
                            for (int j = i; j < entryKey.length; j++) {
                                objectiveHessian[hessianIndexKeyMapping[entryKey[i]][entryKey[j]]] -=
                                        (2 * point[jointTermIndex] * tempProduct + 2 * Math.pow(tempProduct, 2))
                                                / (point[entryKey[i]] * point[entryKey[j]]);
                            }
                        }
                    }
                }
            }
        }
    },
    SCALED_DEPENDENCY {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new UnsupportedOperationException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_ACROSS_DOMAINS {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new UnsupportedOperationException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_ACROSS_DOMAINS {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new UnsupportedOperationException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    L2_NORM {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new UnsupportedOperationException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_AND_L2_NORM {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new UnsupportedOperationException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_AND_L2_NORM {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new UnsupportedOperationException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_ACROSS_DOMAINS_AND_L2_NORM {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new UnsupportedOperationException();
        }
    }, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_ACROSS_DOMAINS_AND_L2_NORM {
        @Override
        public Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
            throw new UnsupportedOperationException();
        }
    }; // TODO: Implement this objective function and add all relevant documentation.

    public abstract Function build(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping);

    /**
     * An interface specifying the methods that all classes defined as possible objective functions for the numerical
     * optimization problem involved in the error rates estimation process, should implement.
     */
    protected interface Function {
        /**
         * Computes the objective computeValue at a particular point. The result is stored in the array passed as argument to
         * this method.
         *
         * @param   point           The point in which to evaluate the objective function.
         * @param   objectiveValue  The array holding objective computeValue to modify.
         */
        void computeValue(double[] point, double[] objectiveValue);

        /**
         * Computes the first derivatives of the objective function at a particular point. The result is stored in the
         * array passed as argument to this method.
         *
         * @param   point               The point in which to evaluate the derivatives.
         * @param   objectiveGradient   The array holding the objective function gradients values to modify.
         */
        void computeGradient(double[] point, double[] objectiveGradient);

        /**
         * Computes the Hessian matrix of the objective function at a particular point. The result is stored in the
         * array passed as argument to this method.
         *
         * @param   point               The point in which to evaluate the Hessian.
         * @param   objectiveHessian    The array holding the Hessian matrix values (in sparse form) to modify.
         */
        void computeHessian(double[] point, double[] objectiveHessian);
    }
}
