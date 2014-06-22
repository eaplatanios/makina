package org.platanios.learn.combination.error;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.primitives.Ints;
import com.ziena.knitro.KnitroJava;
import org.platanios.math.combinatorics.CombinatoricsUtilities;

import java.util.*;

/**
 * Implementation of the numerical optimization problem that needs to be solved in order to estimate error rates of
 * several approximations to a single function, by using only the agreement rates of those functions on an unlabeled set
 * of data, for use with the KNITRO solver.
 *
 * @author Emmanouil Antonios Platanios
 */
class KnitroOptimizationProblem {
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
    /** A mapping used for specifying the sparse format of the Lagrangian Hessian matrix. More specifically, each entry
     * of the two-dimensional array is equal to the corresponding index of that Hessian entry in the sparse,
     * one-dimensional representation of the Hessian matrix. */
    private final int[][] hessianIndexKeyMapping;

    /** The actual KNITRO solver instance. */
    private KnitroJava solver;
    /** Array with the variables with respect to which the optimization is performed. That is, the array with the error
     * rates. */
    private double[] optimizationVariables;
    /** Array with the value of the objective function (it is defined as an array but it holds a single value). */
    private double[] optimizationObjective;
    /** Array with the values of the constraints of the optimization problem (both inequality and equality
     * constraints). */
    private double[] optimizationConstraints;
    /** Array with the values of the objective function derivatives. */
    private double[] optimizationObjectiveGradients;
    /** Array with the values of the constraints derivatives (or, equivalently, the constraints Jacobian). */
    private double[] optimizationConstraintsJacobian;
    /** The Hessian of the Lagrangian. */
    private double[] optimizationHessian;

    /**
     * Initializes all parameters needed for performing the optimization procedure using the KNITRO solver. It also
     * instantiates the solver and makes sure that JVM can "communicate" with the KNITRO solver native libraries.
     *
     * @param   numberOfFunctions   The number of function approximations/classifiers whose error rates we are trying to
     *                              estimate.
     * @param   highestOrder        The highest order of agreement rates to consider and equivalently, the highest order
     *                              of error rates to try and estimate.
     * @param   errorRates          The error rates structure used for this optimization (this structure contains the
     *                              power set indexing information used to index the error rates over all possible power
     *                              sets of functions as a simple one-dimensional array).
     * @param   agreementRates      The agreement rates structure used for this optimization (this structure contains
     *                              the sample agreement rates that are used for defining the equality constraints of
     *                              the problem).
     */
    KnitroOptimizationProblem(int numberOfFunctions,
                              int highestOrder,
                              ErrorRatesPowerSetVector errorRates,
                              AgreementRatesPowerSetVector agreementRates) {
        this.errorRates = errorRates;
        this.agreementRates = agreementRates;

        // Initialize related optimization related variables
        int numberOfVariables = errorRates.length;
        int objectiveGoal = KnitroJava.KTR_OBJGOAL_MINIMIZE;
        int objectiveFunctionType = KnitroJava.KTR_OBJTYPE_GENERAL;
        double[] optimizationStartingPoint = errorRates.array;

        // Set the optimization variables' lower and upper bounds
        double[] variableLowerBounds = new double[numberOfVariables];
        double[] variableUpperBounds = new double[numberOfVariables];

        for (int i = 0; i < numberOfVariables; i++) {
            variableLowerBounds[i] = 0.0;
            variableUpperBounds[i] = 0.5;
        }

        // Set up the optimization constraints
        int numberOfConstraints = agreementRates.indexKeyMapping.size();
        for (int k = 2; k <= highestOrder; k++) {
            numberOfConstraints += CombinatoricsUtilities.getBinomialCoefficient(numberOfFunctions, k) * k;
        }

        int[] constraintTypes = new int[numberOfConstraints];
        for (int k = 0; k < numberOfConstraints; k++) {
            constraintTypes[k] = KnitroJava.KTR_CONTYPE_LINEAR;
        }

        double[] constraintLowerBounds = new double[numberOfConstraints];
        double[] constraintUpperBounds = new double[numberOfConstraints];
        int constraintIndex = 0;
        while(constraintIndex < numberOfConstraints - agreementRates.indexKeyMapping.size()) {
            constraintLowerBounds[constraintIndex] = -KnitroJava.KTR_INFBOUND;
            constraintUpperBounds[constraintIndex++] = 0;
        }

        int agreementRatesIndex = 0;
        while(constraintIndex < numberOfConstraints) {
            constraintLowerBounds[constraintIndex] = agreementRates.array[agreementRatesIndex] - 1;
            constraintUpperBounds[constraintIndex++] = agreementRates.array[agreementRatesIndex++] - 1;
        }

        int numberOfNonZerosInConstraintsJacobian = 0;
        int[][] inner_indexes;
        constraintIndex = 0;

        // Inequality constraints settings
        constraintsJacobian = new ArrayList<Integer[]>();
        ImmutableBiMap.Builder<Integer, int[]> inequalityConstraintsIndexesBuilder
                = new ImmutableBiMap.Builder<Integer, int[]>();
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

        int[] constraintsJacobianConstraintIndexes = new int[numberOfNonZerosInConstraintsJacobian];
        int[] constraintsJacobianVariableIndexes = new int[numberOfNonZerosInConstraintsJacobian];
        constraintIndex = 0;
        for (Integer[] constraintsIndex : constraintsJacobian) {
            constraintsJacobianConstraintIndexes[constraintIndex] = constraintsIndex[0];
            constraintsJacobianVariableIndexes[constraintIndex++] = constraintsIndex[1];
        }

        // Hessian settings
        int numberOfNonZerosInHessian = numberOfVariables * (numberOfVariables + 1) / 2;
        int[] hessianRowIndexes = new int[numberOfVariables * (numberOfVariables + 1) / 2];
        int[] hessianColumnIndexes = new int[numberOfVariables * (numberOfVariables + 1) / 2];
        int[][] hessianIndexKeyMappingTemp = new int[numberOfVariables][numberOfVariables];
        int hessianEntryIndex = 0;
        for (int i = 0; i < numberOfVariables; i++) {
            for (int j = i; j < numberOfVariables; j++) {
                hessianRowIndexes[hessianEntryIndex] = i;
                hessianColumnIndexes[hessianEntryIndex] = j;
                hessianIndexKeyMappingTemp[i][j] = hessianEntryIndex++;
            }
        }
        hessianIndexKeyMapping = hessianIndexKeyMappingTemp;

        // Instantiate the KNITRO solver
        try {
            solver = new KnitroJava();
        } catch (java.lang.Exception  e) {
            System.err.println(e.getMessage());
            return;
        }

        // Configure the KNITRO solver
        if (!solver.setIntParamByName("algorithm", 3)) {
            System.err.println ("Error setting parameter 'algorithm'!");
            return;
        }
        if (!solver.setIntParamByName("blasoption", 1)) {
            System.err.println ("Error setting parameter 'blasoption'!");
            return;
        }
        if (!solver.setDoubleParamByName("feastol", 1.0E-20)) {
            System.err.println ("Error setting parameter 'feastol'!");
            return;
        }
        if (!solver.setIntParamByName("gradopt", 1)) {
            System.err.println ("Error setting parameter 'gradopt'!");
            return;
        }
        if (!solver.setIntParamByName("hessian_no_f", 1)) {
            System.err.println ("Error setting parameter 'hessian_no_f'!");
            return;
        }
        if (!solver.setIntParamByName("hessopt", 1)) {
            System.err.println ("Error setting parameter 'hessopt'!");
            return;
        }
        if (!solver.setIntParamByName("honorbnds", 1)) {
            System.err.println ("Error setting parameter 'honorbnds'!");
            return;
        }
        if (!solver.setIntParamByName("maxit", 100000)) {
            System.err.println ("Error setting parameter 'maxit'!");
            return;
        }
        if (!solver.setDoubleParamByName("opttol", 1E-20)) {
            System.err.println ("Error setting parameter 'opttol'!");
            return;
        }
        if (!solver.setIntParamByName("outlev", 6)) {
            System.err.println ("Error setting parameter 'outlev'!");
            return;
        }
        if (!solver.setIntParamByName("par_numthreads", 1)) {
            System.err.println ("Error setting parameter 'par_numthreads'!");
            return;
        }
        if (!solver.setIntParamByName("scale", 1)) {
            System.err.println ("Error setting parameter 'scale'!");
            return;
        }
        if (!solver.setIntParamByName("soc", 0)) {
            System.err.println ("Error setting parameter 'soc'!");
            return;
        }
        if (!solver.setDoubleParamByName("xtol", 1.0E-20)) {
            System.err.println ("Error setting parameter 'xtol'!");
            return;
        }

        // Initialize the KNITRO solver
        if (!solver.initProblem(numberOfVariables,
                                objectiveGoal,
                                objectiveFunctionType,
                                variableLowerBounds,
                                variableUpperBounds,
                                numberOfConstraints,
                                constraintTypes,
                                constraintLowerBounds,
                                constraintUpperBounds,
                                numberOfNonZerosInConstraintsJacobian,
                                constraintsJacobianVariableIndexes,
                                constraintsJacobianConstraintIndexes,
                                numberOfNonZerosInHessian,
                                hessianRowIndexes,
                                hessianColumnIndexes,
                                optimizationStartingPoint,
                                null)) {
            System.err.println ("Error initializing the problem, "
                    + "KNITRO status = "
                    + solver.getKnitroStatusCode());
            return;
        }

        // Instantiate the global variables used by the KNITRO solver
        optimizationVariables = new double[numberOfVariables];
        optimizationObjective = new double[1];
        optimizationConstraints = new double[numberOfConstraints];
        optimizationObjectiveGradients = new double[numberOfVariables];
        optimizationConstraintsJacobian = new double[numberOfNonZerosInConstraintsJacobian];
        optimizationHessian = new double[numberOfNonZerosInHessian];
    }

    /**
     * Computes the objective value and the constraints values at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the objective function and the constraints.
     * @param   optimizationConstraints The constraints vector to modify.
     * @return                          The objective function value at the given point.
     */
    private double computeObjectiveAndConstraints(double[] optimizationVariables,
                                                  double[] optimizationConstraints) {
        double optimizationObjective = 0;
        int constraintIndex = -1;

        for (BiMap.Entry<List<Integer>, Integer> entry : errorRates.indexKeyMapping.entrySet()) {
            List<Integer> entryKey = entry.getKey();
            int k = entryKey.size();
            if (k > 1) {
                int entryValue = entry.getValue();

                // Objective function
                double tempProduct = 1;
                for (int index : entryKey) {
                    tempProduct *= optimizationVariables[index];
                }
                optimizationObjective += Math.pow(optimizationVariables[entry.getValue()] - tempProduct, 2);

                // Inequality constraints
                for (int inner_index : inequalityConstraintsIndexes.get(entryValue)) {
                    optimizationConstraints[++constraintIndex] =
                            optimizationVariables[errorRates.indexKeyMapping.get(entryKey)]
                                    - optimizationVariables[inner_index];
                }
            }
        }

        int[][] inner_indexes;

        for (BiMap.Entry<List<Integer>, Integer> entry : agreementRates.indexKeyMapping.entrySet()) {
            List<Integer> entryKey = entry.getKey();
            int k = entryKey.size();
            optimizationConstraints[++constraintIndex] =
                    2 * optimizationVariables[errorRates.indexKeyMapping.get(entry.getKey())];
            for (int l = 1; l < k; l++) {
                inner_indexes = CombinatoricsUtilities.getCombinationsOfIntegers(Ints.toArray(entryKey), l);
                for (int[] inner_index : inner_indexes) {
                    optimizationConstraints[constraintIndex] += Math.pow(-1, l)
                            * optimizationVariables[errorRates.indexKeyMapping.get(Ints.asList(inner_index))];
                }
            }
        }

        return optimizationObjective;
    }

    /**
     * Computes the first derivatives of the objective function and the constraints at a particular point.
     *
     * @param   optimizationVariables           The point in which to evaluate the derivatives.
     * @param   optimizationObjectiveGradients  The objective function gradients vector to modify.
     * @param   optimizationConstraintsJacobian The constraints Jacobian (in sparse/vector form) to modify.
     */
    private void computeGradients(double[] optimizationVariables,
                                  double[] optimizationObjectiveGradients,
                                  double[] optimizationConstraintsJacobian) {
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

        int i = 0;

        for (Integer[] constraintsIndex : constraintsJacobian) {
            optimizationConstraintsJacobian[i++] = constraintsIndex[2];
        }
    }

    /**
     * Computes the Hessian of the Lagrangian at a particular point. The constraints in this case are linear and so they
     * do not contribute to the Hessian value.
     *
     * @param   optimizationVariables   The point in which to evaluate the Hessian.
     * @param   optimizationHessian     The Hessian (in sparse/vector form) to modify.
     * @param   onlyConstraints         Variable indicating whether to compute the Hessian only for the constraints, or
     *                                  for the whole Lagrangian function.
     */
    private void computeHessian(double[] optimizationVariables,
                                double[] optimizationHessian,
                                boolean onlyConstraints) {
        for (int i = 0; i < optimizationHessian.length; i++) {
            optimizationHessian[i] = 0;
        }

        if (onlyConstraints) {
            return;
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

    /**
     * Solves the numerical optimization problem and returns the error rates estimates in {@code double[]} format.
     *
     * @return  The error rates estimates in a {@code double[]} format.
     */
    double[] solve() {
        // Solve the optimization problem using KNITRO and record its status code
        int  knitroStatusCode;
        do {
            knitroStatusCode = solver.solve(0,
                                            optimizationObjective,
                                            optimizationConstraints,
                                            optimizationObjectiveGradients,
                                            optimizationConstraintsJacobian,
                                            optimizationHessian);

            switch (knitroStatusCode) {
                case KnitroJava.KTR_RC_EVALFC:
                    optimizationVariables = solver.getCurrentX();
                    optimizationObjective[0] = computeObjectiveAndConstraints(optimizationVariables,
                            optimizationConstraints);
                    break;
                case KnitroJava.KTR_RC_EVALGA:
                    optimizationVariables = solver.getCurrentX();
                    computeGradients(optimizationVariables,
                            optimizationObjectiveGradients,
                            optimizationConstraintsJacobian);
                    break;
                case KnitroJava.KTR_RC_EVALH:
                    optimizationVariables = solver.getCurrentX();
                    computeHessian(optimizationVariables, optimizationHessian, false);
                    break;
                case KnitroJava.KTR_RC_EVALH_NO_F:
                    optimizationVariables = solver.getCurrentX();
                    computeHessian(optimizationVariables, optimizationHessian, true);
                    break;
            }
        }
        while (knitroStatusCode > 0);

        // Display the KNITRO status after completing the optimization procedure
        System.out.print ("KNITRO optimization finished! Status " + knitroStatusCode + ": ");
        switch (knitroStatusCode) {
            case KnitroJava.KTR_RC_OPTIMAL:
                System.out.println ("Converged to optimality!");
                break;
            case KnitroJava.KTR_RC_ITER_LIMIT:
                System.out.println ("Reached the maximum number of allowed iterations!");
                break;
            case KnitroJava.KTR_RC_NEAR_OPT:
            case KnitroJava.KTR_RC_FEAS_XTOL:
            case KnitroJava.KTR_RC_FEAS_FTOL:
            case KnitroJava.KTR_RC_FEAS_NO_IMPROVE:
                System.out.println ("Could not improve upon the current iterate!");
                break;
            case KnitroJava.KTR_RC_TIME_LIMIT:
                System.out.println ("Reached the maximum CPU time allowed!");
                break;
            default:
                System.out.println ("Failed!");
        }

        // Destroy the KNITRO native object
        solver.destroyInstance();

        return optimizationVariables;
    }
}
