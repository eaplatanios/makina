package org.platanios.learn.classification.reflection.perception;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.primitives.Ints;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
    /** Logger object used by this class. */
    private static final Logger logger = LogManager.getLogger("Error Rates Estimation / IpOpt Optimization");

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
        // Output Settings
        if (!setIntegerOption("print_level", 5)) { // Must be between 0 and 12.
            logger.error("Error setting parameter 'print_level'!");
            return;
        }
        // Termination Settings
        if (!setNumericOption("tol", 1e-8)) {
            logger.error("Error setting parameter 'tol'!");
            return;
        }
        if (!setIntegerOption("max_iter", 3000)) {
            logger.error("Error setting parameter 'max_iter'!");
            return;
        }
        if (!setNumericOption("max_cpu_time", 1000000)) {
            logger.error("Error setting parameter 'max_cpu_time'!");
            return;
        }
        if (!setNumericOption("dual_inf_tol", 1)) {
            logger.error("Error setting parameter 'dual_inf_tol'!");
            return;
        }
        if (!setNumericOption("constr_viol_tol", 1e-15)) {
            logger.error("Error setting parameter 'constr_viol_tol'!");
            return;
        }
        if (!setNumericOption("compl_inf_tol", 1e-4)) {
            logger.error("Error setting parameter 'compl_inf_tol'!");
            return;
        }
        // NLP Scaling Settings
        if (!setNumericOption("obj_scaling_factor", 1)) {
            logger.error("Error setting parameter 'obj_scaling_factor'!");
            return;
        }
        if (!setStringOption("nlp_scaling_method", "equilibration-based")) {
            logger.error("Error setting parameter 'nlp_scaling_method'!");
            return;
        }
        if (!setNumericOption("nlp_scaling_max_gradient", 100)) {
            logger.error("Error setting parameter 'nlp_scaling_max_gradient'!");
            return;
        }
        if (!setNumericOption("nlp_scaling_min_value", 1e-8)) {
            logger.error("Error setting parameter 'nlp_scaling_min_value'!");
            return;
        }
        // NLP Settings
        if (!setNumericOption("bound_relax_factor", 0)) {
            logger.error("Error setting parameter 'bound_relax_factor'!");
            return;
        }
        if (!setStringOption("honor_original_bounds", "yes")) {
            logger.error("Error setting parameter 'honor_original_bounds'!");
            return;
        }
        if (!setStringOption("jac_c_constant", "yes")) {
            logger.error("Error setting parameter 'jac_c_constant'!");
            return;
        }
        if (!setStringOption("jac_d_constant", "yes")) {
            logger.error("Error setting parameter 'jac_d_constant'!");
            return;
        }
        if (!setStringOption("hessian_constant", "no")) {
            logger.error("Error setting parameter 'hessian_constant'!");
            return;
        }
        // Barrier Parameter Settings
        if (!setStringOption("mehrotra_algorithm", "no")) {
            logger.error("Error setting parameter 'mehrotra_algorithm'!");
            return;
        }
        if (!setStringOption("mu_strategy", "adaptive")) {
            logger.error("Error setting parameter 'mu_strategy'!");
            return;
        }
        if (!setStringOption("mu_oracle", "quality-function")) {
            logger.error("Error setting parameter 'mu_oracle'!");
            return;
        }
        if (!setStringOption("fixed_mu_oracle", "average_compl")) {
            logger.error("Error setting parameter 'fixed_mu_oracle'!");
            return;
        }
        // Restoration Phase Settings
        if (!setStringOption("expect_infeasible_problem", "no")) {
            logger.error("Error setting parameter 'expect_infeasible_problem'!");
            return;
        }
        // Linear Solver Settings
        if (!setStringOption("linear_solver", "ma27")) {
            logger.error("Error setting parameter 'linear_solver'!");
            return;
        }
        if (!setStringOption("linear_system_scaling", "mc19")) {
            logger.error("Error setting parameter 'linear_system_scaling'!");
            return;
        }
        if (!setStringOption("linear_scaling_on_demand", "yes")) {
            logger.error("Error setting parameter 'linear_scaling_on_demand'!");
            return;
        }
    }

    /**
     * Sets the variables bounds and the constraints bounds for the optimization problem. The bounds are stored in
     * arrays passed as arguments to this method.
     *
     * @param   numberOfVariables       The number of variables in the optimization problem.
     * @param   variableLowerBounds     The array holding the variables lower bounds values to modify.
     * @param   variableUpperBounds     The array holding the variables upper bounds values to modify.
     * @param   numberOfConstraints     The number of constraints in the optimization problem.
     * @param   constraintsLowerBounds  The array holding the constraints lower bounds values to modify.
     * @param   constraintsUpperBounds  The array holding the constraints upper bounds values to modify.
     * @return                          A boolean value indicating whether the method execution was successful or not.
     */
    @Override
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

    /**
     * Sets the starting point used by the IpOpt nonlinear solver. The starting point is stored in an array passed as an
     * argument to this method.
     *
     * @param   numberOfVariables           The number of variables in the optimization problem.
     * @param   initializePoint             Boolean value indicating whether this method must provide values for the
     *                                      optimization starting point.
     * @param   point                       The array holding the starting point value to modify.
     * @param   initializeBoundsMultipliers Boolean value indicating whether this method must provide initial values for
     *                                      the bounds multipliers.
     * @param   lowerBoundsMultipliers      The array holding the lower bounds multipliers values to modify.
     * @param   upperBoundsMultipliers      The array holding the upper bounds multipliers values to modify.
     * @param   numberOfConstraints         The number of constraints in the optimization problem.
     * @param   initializeLambda            Boolean value indicating whether this method must provide initial values for
     *                                      the Lagrange multipliers.
     * @param   lambda                      The array holding the Lagrange multipliers values to modify.
     * @return                              A boolean value indicating whether the method execution was successful or
     *                                      not.
     */
    @Override
    protected boolean get_starting_point(int numberOfVariables,
                                         boolean initializePoint,
                                         double[] point,
                                         boolean initializeBoundsMultipliers,
                                         double[] lowerBoundsMultipliers,
                                         double[] upperBoundsMultipliers,
                                         int numberOfConstraints,
                                         boolean initializeLambda,
                                         double[] lambda)
    {
        if(initializePoint)
            point = startingPoint;
        return true;
    }

    /**
     * Computes the objective value at a particular point. The result is stored in an array passed as an argument to
     * this method.
     *
     * @param   numberOfVariables   The number of variables in the optimization problem.
     * @param   point               The point in which to evaluate the objective function.
     * @param   newPoint            Boolean value that is equal to false if any evaluation method was previously called
     *                              with the same point, and true otherwise.
     * @param   objectiveValue      The array holding objective value to modify.
     * @return                      A boolean value indicating whether the method execution was successful or not.
     */
    @Override
    protected boolean eval_f(int numberOfVariables,
                             double[] point,
                             boolean newPoint,
                             double[] objectiveValue) {
        objectiveFunction.computeObjective(point, objectiveValue);
        return true;
    }

    /**
     * Computes the first derivatives of the objective function at a particular point. The result is stored in an array
     * passed as an argument to this method.
     *
     * @param   numberOfVariables   The number of variables in the optimization problem.
     * @param   point               The point in which to evaluate the objective function.
     * @param   newPoint            Boolean value that is equal to false if any evaluation method was previously called
     *                              with the same point, and true otherwise.
     * @param   objectiveGradient   The array holding the objective function gradients values to modify.
     * @return                      A boolean value indicating whether the method execution was successful or not.
     */
    @Override
    protected boolean eval_grad_f(int numberOfVariables,
                                  double[] point,
                                  boolean newPoint,
                                  double[] objectiveGradient) {
        objectiveFunction.computeGradient(point, objectiveGradient);
        return true;
    }

    /**
     * Computes the Hessian matrix of the Lagrange function at a particular point. The result is stored in an array
     * passed as an argument to this method. If that array, which is passed as a parameter, is null, then the structure
     * of the Hessian matrix (i.e., the indexes of the non-zero entries in the matrix) is returned instead.
     *
     * @param   numberOfVariables           The number of variables in the optimization problem.
     * @param   point                       The point in which to evaluate the objective function.
     * @param   newPoint                    Boolean value that is equal to false if any evaluation method was previously
     *                                      called with the same point, and true otherwise.
     * @param   objectiveScalingFactor      Scaling factor multiplying the objective function within the Lagrange
     *                                      function.
     * @param   numberOfConstraints         The number of constraints in the optimization problem.
     * @param   lambda                      The values of the Lagrange multipliers.
     * @param   newLambda                   Boolean value that is equal to false if any evaluation method was previously
     *                                      called with the same Lagrange multipliers, and true otherwise.
     * @param   numberOfNonZerosInHessian   The number of nonzero elements in the Hessian matrix.
     * @param   rowIndexes                  The row indexes of the nonzero elements in the Hessian matrix.
     * @param   columnIndexes               The column indexes of the nonzero elements in the Hessian matrix.
     * @param   objectiveHessian            The array holding the Hessian matrix values (in sparse form) to modify.
     * @return                              A boolean value indicating whether the method execution was successful or
     *                                      not.
     */
    @Override
    protected boolean eval_h(int numberOfVariables,
                             double[] point,
                             boolean newPoint,
                             double objectiveScalingFactor,
                             int numberOfConstraints,
                             double[] lambda,
                             boolean newLambda,
                             int numberOfNonZerosInHessian,
                             int[] rowIndexes,
                             int[] columnIndexes,
                             double[] objectiveHessian) {
        if (objectiveHessian == null) {
            int hessianEntryIndex = 0;
            for (int i = 0; i < numberOfVariables; i++) {
                for (int j = i; j < numberOfVariables; j++) {
                    rowIndexes[hessianEntryIndex] = i;
                    columnIndexes[hessianEntryIndex] = j;
                }
            }
        } else {
            objectiveFunction.computeHessian(point, objectiveHessian);
        }
        return true;
    }

    /**
     * Computes the constraints values at a particular point. The result is stored in an array passed as an argument to
     * this method.
     *
     * @param   numberOfVariables   The number of variables in the optimization problem.
     * @param   point               The point in which to evaluate the objective function.
     * @param   newPoint            Boolean value that is equal to false if any evaluation method was previously called
     *                              with the same point, and true otherwise.
     * @param   numberOfConstraints The number of constraints in the optimization problem.
     * @param   constraints         The array holding the constraints values to modify.
     * @return                      A boolean value indicating whether the method execution was successful or not.
     */
    @Override
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
     * this case and so their derivatives values are constant). The result is stored in an array passed as an argument
     * to this method. If that array, which is passed as a parameter, is null, then the structure of the Jacobian matrix
     * (i.e., the indexes of the non-zero entries in the matrix) is returned instead.
     *
     * @param   numberOfVariables           The number of variables in the optimization problem.
     * @param   point                       The point in which to evaluate the objective function.
     * @param   newPoint                    Boolean value that is equal to false if any evaluation method was previously
     *                                      called with the same point, and true otherwise.
     * @param   numberOfConstraints         The number of constraints in the optimization problem.
     * @param   numberOfNonZerosInJacobian  The number of nonzero elements in the Jacobian matrix.
     * @param   rowIndexes                  The row indexes of the nonzero elements in the Jacobian matrix.
     * @param   columnIndexes               The column indexes of the nonzero elements in the Jacobian matrix.
     * @param   constraintsGradients        The array holding the constraints Jacobian matrix values (in sparse form) to
     *                                      modify.
     * @return                              A boolean value indicating whether the method execution was successful or
     *                                      not.
     */
    @Override
    protected boolean eval_jac_g(int numberOfVariables,
                                 double[] point,
                                 boolean newPoint,
                                 int numberOfConstraints,
                                 int numberOfNonZerosInJacobian,
                                 int[] rowIndexes,
                                 int[] columnIndexes,
                                 double[] constraintsGradients) {
        if (constraintsGradients == null) {
            int constraintIndex = 0;
            for (Integer[] constraintsIndex : constraintsJacobian) {
                rowIndexes[constraintIndex] = constraintsIndex[0];
                columnIndexes[constraintIndex++] = constraintsIndex[1];
            }
        } else {
            int i = 0;
            for (Integer[] constraintsIndex : constraintsJacobian)
                constraintsGradients[i++] = constraintsIndex[2];
        }
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public double[] solve() {
        OptimizeNLP();
        return getVariableValues();
    }
}