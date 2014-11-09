package org.platanios.learn.classification.reflection.perception;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.primitives.Ints;
import org.coinor.Ipopt;
import org.platanios.learn.math.combinatorics.CombinatoricsUtilities;

import java.util.*;

/**
 * Implementation of the numerical optimization problem that needs to be solved in order to estimate error rates of
 * several approximations to a single function, by using only the agreement rates of those functions on an unlabeled set
 * of data, and that uses the IpOpt solver.
 *
 * @author Emmanouil Antonios Platanios
 */
class OptimizationProblemIpOpt extends Ipopt implements OptimizationProblem {
    /** The error rates structure used for this optimization (this structure contains the power set indexing
     * information used to index the error rates over all possible power sets of functions as a simple one-dimensional
     * array). */
    private final ErrorRatesPowerSetVector errorRates;
    /** The agreement rates structure used for this optimization (this structure contains the sample agreement
     * rates that are used for defining the equality constraints of the problem). */
    private final AgreementRatesPowerSetVector agreementRates;
    /** A map containing the error rates array indices that correspond to each inequality constraint. It is used in
     * order to make the inequality constraints calculation fast. */
    private final BiMap<Integer, int[]> inequalityConstraintsIndexes;
    /** A list where each entry is an array with three elements: (i) a constraint index, (ii) an error rate index and
     * (iii) the derivative of the constraint specified in (i) with respect to the error rate specified in (ii). It is
     * used to make the constraints Jacobian calculation fast. */
    private final List<Integer[]> constraintsJacobian;
    /** An instance of the class containing methods that compute the objective function that we wish to minimize, its
     * gradients with respect to the optimization variables and its Hessian with respect to the optimization
     * variables. */
    private final ObjectiveFunction objectiveFunction;
    /** The starting point of the numerical optimization procedure. */
    private final double[] startingPoint;

    /**
     * Initializes all parameters needed for performing the optimization procedure using the IpOpt solver. It also
     * initializes the solver and makes sure that JVM can "communicate" with the IpOpt solver native libraries.
     *
     * @param   numberOfFunctions       The number of function approximations/classifiers whose error rates we want to
     *                                  estimate.
     * @param   highestOrder            The highest order of agreement rates to consider and equivalently, the highest
     *                                  order of error rates to try and estimate.
     * @param   errorRates              The error rates structure used for this optimization (this structure contains
     *                                  the power set indexing information used to index the error rates over all
     *                                  possible power sets of functions as a simple one-dimensional array).
     * @param   agreementRates          The agreement rates structure used for this optimization (this structure
     *                                  contains the sample agreement rates that are used for defining the equality
     *                                  constraints of the problem).
     * @param   objectiveFunctionType   The type of objective function to minimize (e.g. minimize dependency, scaled
     *                                  dependency, etc.).
     */
    OptimizationProblemIpOpt(int numberOfFunctions,
                             int highestOrder,
                             ErrorRatesPowerSetVector errorRates,
                             AgreementRatesPowerSetVector agreementRates,
                             ObjectiveFunctionType objectiveFunctionType) {
        this.errorRates = errorRates;
        this.agreementRates = agreementRates;
        startingPoint = errorRates.array;

        // Compute the number of variables in the optimization problem and the number of the constraints
        int numberOfVariables = errorRates.length;
        int numberOfConstraints = agreementRates.indexKeyMapping.size();
        for (int k = 2; k <= highestOrder; k++)
            numberOfConstraints += CombinatoricsUtilities.getBinomialCoefficient(numberOfFunctions, k) * k;

        int numberOfNonZerosInConstraintsJacobian = 0;
        int[][] inner_indexes;
        int constraintIndex = 0;

        // Inequality constraints settings
        constraintsJacobian = new ArrayList<>();
        ImmutableBiMap.Builder<Integer, int[]> inequalityConstraintsIndexesBuilder = new ImmutableBiMap.Builder<>();
        for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
            List<Integer> entryKey = entry.getKey();
            int k = entryKey.size();
            if (entry.getKey().size() > 1) {
                int entryValue = entry.getValue();
                int jointErrorRatesIndex = errorRates.indexKeyMapping.get(entryKey);
                inner_indexes = CombinatoricsUtilities.getCombinationsOfIntegers(Ints.toArray(entryKey), k - 1);
                int[] inner_keys = new int[inner_indexes.length];
                for (int i = 0; i < inner_indexes.length; i++) {
                    inner_keys[i] = errorRates.indexKeyMapping.get(Ints.asList(inner_indexes[i]));
                    constraintsJacobian.add(new Integer[] { constraintIndex, jointErrorRatesIndex, 1 });
                    constraintsJacobian.add(new Integer[] { constraintIndex++, inner_keys[i], -1 });
                }
                inequalityConstraintsIndexesBuilder.put(entryValue, inner_keys);
                numberOfNonZerosInConstraintsJacobian += 2 * inner_indexes.length;
            }
        }
        inequalityConstraintsIndexes = inequalityConstraintsIndexesBuilder.build();

        // Equality constraints settings
        for (BiMap.Entry<List<Integer>, Integer> entry : agreementRates.indexKeyMapping.entrySet()) {
            List<Integer> entryKey = entry.getKey();
            int k = entryKey.size();
            constraintsJacobian.add(new Integer[] {
                    constraintIndex,
                    errorRates.indexKeyMapping.get(entryKey),
                    2
            });
            numberOfNonZerosInConstraintsJacobian += 1;
            for (int l = 1; l < k; l++) {
                inner_indexes = CombinatoricsUtilities.getCombinationsOfIntegers(Ints.toArray(entryKey), l);
                for (int[] inner_index : inner_indexes) {
                    constraintsJacobian.add(new Integer[] {
                            constraintIndex,
                            errorRates.indexKeyMapping.get(Ints.asList(inner_index)),
                            (int)Math.pow(-1, l)
                    });
                }
                numberOfNonZerosInConstraintsJacobian += inner_indexes.length;
            }
            constraintIndex++;
        }

        // Hessian settings
        int numberOfNonZerosInHessian = numberOfVariables * (numberOfVariables + 1) / 2;
        int[][] hessianIndexKeyMapping = new int[numberOfVariables][numberOfVariables];
        int hessianEntryIndex = 0;
        for (int i = 0; i < numberOfVariables; i++)
            for (int j = i; j < numberOfVariables; j++)
                hessianIndexKeyMapping[i][j] = hessianEntryIndex++;

        // Instantiate the chosen objective function class
        objectiveFunction = objectiveFunctionType.build(errorRates, hessianIndexKeyMapping);

        // Initialize the IpOpt solver
        create(numberOfVariables,
               numberOfConstraints,
               numberOfNonZerosInConstraintsJacobian,
               numberOfNonZerosInHessian,
               Ipopt.C_STYLE);

        // Configure the IpOpt solver
        System.out.println(setStringOption("linear_solver", "ma97"));
    }

    protected boolean get_bounds_info(int numberOfVariables,
                                      double[] variableLowerBounds,
                                      double[] variableUpperBounds,
                                      int numberOfConstraints,
                                      double[] constraintsLowerBounds,
                                      double[] constraintsUpperBounds)
    {
        // Set the variable bounds
        for (int i = 0; i < numberOfVariables; i++) {
            variableLowerBounds[i] = 0.0;
            variableUpperBounds[i] = 0.5;
        }
        // Set the constraint bounds
        int constraintIndex = 0;
        while(constraintIndex < numberOfConstraints - agreementRates.indexKeyMapping.size()) {
            constraintsLowerBounds[constraintIndex] = Double.NEGATIVE_INFINITY;
            constraintsUpperBounds[constraintIndex++] = 0;
        }
        int agreementRatesIndex = 0;
        while(constraintIndex < numberOfConstraints) {
            constraintsLowerBounds[constraintIndex] = agreementRates.array[agreementRatesIndex] - 1;
            constraintsUpperBounds[constraintIndex++] = agreementRates.array[agreementRatesIndex++] - 1;
        }
        return true;
    }

    protected boolean get_starting_point(int numberOfVariables,
                                         boolean initializePoint,
                                         double[] point,
                                         boolean init_z,
                                         double[] z_L,
                                         double[] z_U,
                                         int numberOfConstraints,
                                         boolean initializeLambda,
                                         double[] lambda)
    {
        if(initializePoint)
            point = startingPoint;
        return true;
    }

    /**
     * Computes the value of the objective function at a particular point.
     */
    protected boolean eval_f(int numberOfVariables,
                             double[] point,
                             boolean newPoint,
                             double[] objectiveValue) {
        objectiveFunction.computeObjective(point, objectiveValue);
        return true;
    }

    /**
     * Computes the gradient of the objective function at a particular point.
     */
    protected boolean eval_grad_f(int numberOfVariables,
                                  double[] point,
                                  boolean newPoint,
                                  double[] objectiveGradient) {
        objectiveFunction.computeGradient(point, objectiveGradient);
        return true;
    }

    /**
     * Computes the Hessian of the objective function at a particular point.
     */
    protected boolean eval_h(int numberOfVariables,
                             double[] point,
                             boolean newPoint,
                             double objectiveScalingFactor,
                             int numberOfConstraints,
                             double[] lambda,
                             boolean newLambda,
                             int numberOfNonZerosInHessian,
                             int[] iRow,
                             int[] jCol,
                             double[] objectiveHessian) {
        if (objectiveHessian == null) {
            int hessianEntryIndex = 0;
            for (int i = 0; i < numberOfVariables; i++) {
                for (int j = i; j < numberOfVariables; j++) {
                    iRow[hessianEntryIndex] = i;
                    jCol[hessianEntryIndex] = j;
                }
            }
        } else {
            objectiveFunction.computeHessian(point, objectiveHessian);
        }
        return true;
    }

    /**
     * Computes the constraints values at a particular point.
     */
    protected boolean eval_g(int numberOfVariables,
                             double[] point,
                             boolean newPoint,
                             int numberOfConstraints,
                             double[] constraints) {
        int constraintIndex = -1;
        for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
            List<Integer> entryKey = entry.getKey();
            int k = entryKey.size();
            if (k > 1)
                for (int inner_index : inequalityConstraintsIndexes.get(entry.getValue()))
                    constraints[++constraintIndex] =
                            point[errorRates.indexKeyMapping.get(entryKey)] - point[inner_index];
        }
        int[][] inner_indexes;
        for (BiMap.Entry<List<Integer>, Integer> entry : agreementRates.indexKeyMapping.entrySet()) {
            List<Integer> entryKey = entry.getKey();
            int k = entryKey.size();
            constraints[++constraintIndex] = 2 * point[errorRates.indexKeyMapping.get(entry.getKey())];
            for (int l = 1; l < k; l++) {
                inner_indexes = CombinatoricsUtilities.getCombinationsOfIntegers(Ints.toArray(entryKey), l);
                for (int[] inner_index : inner_indexes)
                    constraints[constraintIndex] +=
                            Math.pow(-1, l) * point[errorRates.indexKeyMapping.get(Ints.asList(inner_index))];
            }
        }
        return true;
    }

    /**
     * Computes the first derivatives of the constraints at a particular point (the constraints are actually linear in
     * this case and so their derivatives values are constant). If the values array which is passed as a parameter is
     * null, then the structure of the Jacobian matrix (i.e., the indexes of the non-zero entries in the matrix) is
     * returned instead.
     */
    protected boolean eval_jac_g(int numberOfVariables,
                                 double[] point,
                                 boolean newPoint,
                                 int numberOfConstraints,
                                 int numberOfNonZerosInConstraintsJacobian,
                                 int[] iRow,
                                 int[] jCol,
                                 double[] constraintsGradients) {
        if (constraintsGradients == null) {
            int constraintIndex = 0;
            for (Integer[] constraintsIndex : constraintsJacobian) {
                iRow[constraintIndex] = constraintsIndex[0];
                jCol[constraintIndex++] = constraintsIndex[1];
            }
        } else {
            int i = 0;
            for (Integer[] constraintsIndex : constraintsJacobian)
                constraintsGradients[i++] = constraintsIndex[2];
        }
        return true;
    }

    /**
     * Solves the numerical optimization problem and returns the error rates estimates in {@code double[]} format.
     *
     * @return  The error rates estimates in a {@code double[]} format.
     */
    public double[] solve() {
        OptimizeNLP();
        return getVariableValues();
    }
}
