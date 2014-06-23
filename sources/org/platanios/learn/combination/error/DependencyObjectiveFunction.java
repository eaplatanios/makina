package org.platanios.learn.combination.error;

import com.google.common.collect.BiMap;
import com.google.common.primitives.Ints;

import java.util.List;

/**
 * An implementation of an objective function that quantifies the dependencies between the error rates of the  different
 * functions. The goal of the optimization procedure using this objective function is to minimize it, subject to some
 * equality and some inequality constraints, and obtain an estimate for the error rates.
 *
 * @author Emmanouil Antonios Platanios
 */
public class DependencyObjectiveFunction implements ObjectiveFunction {
    /** The error rates structure used for all calculations (this structure contains the power set indexing information
     * used to index the error rates over all possible power sets of functions as a simple one-dimensional array). */
    private final ErrorRatesPowerSetVector errorRates;
    /** A mapping used for specifying the sparse format of the Lagrangian Hessian matrix. More specifically, each entry
     * of the two-dimensional array is equal to the corresponding index of that Hessian entry in the sparse,
     * one-dimensional, representation of the Hessian matrix. */
    private final int[][] hessianIndexKeyMapping;

    /**
     * Initializes all parameters needed for computing the objective function value, its derivatives with respect to its
     * input variables and its Hessian with respect to the input variables.
     *
     * @param   errorRates              The error rates structure used for all calculations (this structure contains the
     *                                  power set indexing information used to index the error rates over all possible
     *                                  power sets of functions as a simple one-dimensional array).
     * @param   hessianIndexKeyMapping  A mapping used for specifying the sparse format of the Lagrangian Hessian
     *                                  matrix. More specifically, each entry of the two-dimensional array is equal to
     *                                  the corresponding index of that Hessian entry in the sparse, one-dimensional,
     *                                  representation of the Hessian matrix.
     */
    public DependencyObjectiveFunction(ErrorRatesPowerSetVector errorRates, int[][] hessianIndexKeyMapping) {
        this.errorRates = errorRates;
        this.hessianIndexKeyMapping = hessianIndexKeyMapping;
    }

    /**
     * Computes the objective value and the constraints values at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the objective function and the constraints.
     * @param   optimizationObjective   The objective value to set for the given point.
     */
    public void computeObjective(double[] optimizationVariables,
                                 double[] optimizationObjective) {
        optimizationObjective[0] = 0;

        for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
            List<Integer> entryKey = entry.getKey();
            int k = entryKey.size();
            if (k > 1) {
                // Objective function
                double tempProduct = 1;
                for (int index : entryKey) {
                    tempProduct *= optimizationVariables[index];
                }
                optimizationObjective[0] += Math.pow(optimizationVariables[entry.getValue()] - tempProduct, 2);
            }
        }
    }

    /**
     * Computes the first derivatives of the objective function and the constraints at a particular point.
     *
     * @param   optimizationVariables           The point in which to evaluate the derivatives.
     * @param   optimizationObjectiveGradients  The objective function gradients vector to modify.
     */
    public void computeGradient(double[] optimizationVariables,
                                double[] optimizationObjectiveGradients) {
        for (int i = 0; i < optimizationObjectiveGradients.length; i++) {
            optimizationObjectiveGradients[i] = 0;
        }

        for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
            if (entry.getKey().size() > 1) {
                double tempProduct = 1;
                List<Integer> indexes = entry.getKey();
                for (int index : indexes) {
                    tempProduct *= optimizationVariables[index];
                }
                double term = optimizationVariables[entry.getValue()] - tempProduct;
                optimizationObjectiveGradients[entry.getValue()] += 2 * term;
                for (int index : indexes) {
                    optimizationObjectiveGradients[index] -= 2 * term * tempProduct / optimizationVariables[index];
                }
            }
        }
    }

    /**
     * Computes the Hessian of the Lagrangian at a particular point. The constraints in this case are linear and so they
     * do not contribute to the Hessian value.
     *
     * @param   optimizationVariables   The point in which to evaluate the Hessian.
     * @param   optimizationHessian     The Hessian (in sparse/vector form) to modify.
     */
    public void computeHessian(double[] optimizationVariables,
                               double[] optimizationHessian) {
        for (int i = 0; i < optimizationHessian.length; i++) {
            optimizationHessian[i] = 0;
        }

        for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
            int[] entryKey = Ints.toArray(entry.getKey());
            if (entryKey.length > 1) {
                double tempProduct = 1;
                for (int index : entryKey) {
                    tempProduct *= optimizationVariables[index];
                }
                int jointTermIndex = entry.getValue();
                optimizationHessian[hessianIndexKeyMapping[jointTermIndex][jointTermIndex]] = 2;
                for (int i = 0; i < entryKey.length; i++) {
                    optimizationHessian[hessianIndexKeyMapping[entryKey[i]][jointTermIndex]] =
                            2 * tempProduct / optimizationVariables[entryKey[i]];
                    optimizationHessian[hessianIndexKeyMapping[entryKey[i]][entryKey[i]]] +=
                            2 * optimizationVariables[jointTermIndex] * tempProduct
                                    / Math.pow(optimizationVariables[entryKey[i]], 2);

                    // Notice that the (map) keys of indexKeyMapping are in ascending order by construction
                    for (int j = i; j < entryKey.length; j++) {
                        optimizationHessian[hessianIndexKeyMapping[entryKey[i]][entryKey[j]]] -=
                                (2 * optimizationVariables[jointTermIndex] * tempProduct + 2 * Math.pow(tempProduct, 2))
                                        / (optimizationVariables[entryKey[i]] * optimizationVariables[entryKey[j]]);
                    }
                }
            }
        }
    }
}
