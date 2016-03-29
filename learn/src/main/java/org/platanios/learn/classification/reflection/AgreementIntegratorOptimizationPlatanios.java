package org.platanios.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.primitives.Ints;
import com.ziena.knitro.KnitroJava;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.math.CombinatoricsUtilities;
import org.platanios.optimization.NonlinearInteriorPointSolver;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class AgreementIntegratorOptimizationPlatanios implements AgreementIntegratorOptimization {
    /** Logger object used by this class. */
    private static final Logger logger = LogManager.getLogger("Error Rates Estimation / Platanios Optimization");

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

    /** Nonlinear optimization solver object used by this class. */
    private NonlinearInteriorPointSolver solver;
    /** An instance of the class containing methods that compute the objective function that we wish to minimize, its
     * gradients with respect to the optimization variables and its Hessian with respect to the optimization
     * variables. */
    private AgreementIntegratorObjective.Function objectiveFunction;
    /** Array with the variables with respect to which the optimization is performed. That is, the array with the error
     * rates. */
    private double[] point;
    /** Array with the computeValue of the objective function (it is defined as an array but it holds a single computeValue). */
    private double[] objectiveValue;
    /** Array with the values of the constraints of the optimization problem (both inequality and equality
     * constraints). */
    private double[] constraints;
    /** Array with the values of the objective function derivatives. */
    private double[] objectiveGradient;
    /** Array with the values of the constraints derivatives (or, equivalently, the constraints Jacobian). */
    private double[] constraintsGradients;
    /** The Hessian of the Lagrangian. */
    private double[] objectiveHessian;

    /**
     * Initializes all parameters needed for performing the optimization procedure using the Platanios solver. It also
     * instantiates the solver.
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
    AgreementIntegratorOptimizationPlatanios(int numberOfFunctions,
                                             int highestOrder,
                                             ErrorRatesPowerSetVector errorRates,
                                             AgreementRatesPowerSetVector agreementRates,
                                             AgreementIntegratorObjective objectiveFunctionType) {
        this.errorRates = errorRates;
        this.agreementRates = agreementRates;

        // Initialize related optimization related variables
        int numberOfVariables = errorRates.length;
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

        double[] constraintsLowerBounds = new double[numberOfConstraints];
        double[] constraintsUpperBounds = new double[numberOfConstraints];
        int constraintIndex = 0;
        while (constraintIndex < numberOfConstraints - agreementRates.indexKeyMapping.size()) {
            constraintsLowerBounds[constraintIndex] = -KnitroJava.KTR_INFBOUND;
            constraintsUpperBounds[constraintIndex++] = 0;
        }

        int agreementRatesIndex = 0;
        while (constraintIndex < numberOfConstraints) {
            constraintsLowerBounds[constraintIndex] = agreementRates.array[agreementRatesIndex] - 1;
            constraintsUpperBounds[constraintIndex++] = agreementRates.array[agreementRatesIndex++] - 1;
        }

        int numberOfNonZerosInConstraintsJacobian = 0;
        int[][] inner_indexes;
        constraintIndex = 0;

        // Inequality constraints settings
        constraintsJacobian = new ArrayList<>();
        ImmutableBiMap.Builder<Integer, int[]> inequalityConstraintsIndexesBuilder
                = new ImmutableBiMap.Builder<>();
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
                    constraintsJacobian.add(new Integer[]{constraintIndex, jointErrorRatesIndex, 1});
                    constraintsJacobian.add(new Integer[]{constraintIndex++, inner_keys[i], -1});
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
            constraintsJacobian.add(new Integer[]{
                    constraintIndex,
                    errorRates.indexKeyMapping.get(entryKey),
                    2
            });
            numberOfNonZerosInConstraintsJacobian += 1;
            for (int l = 1; l < k; l++) {
                inner_indexes = CombinatoricsUtilities.getCombinationsOfIntegers(Ints.toArray(entryKey), l);
                for (int[] inner_index : inner_indexes) {
                    constraintsJacobian.add(new Integer[]{
                            constraintIndex,
                            errorRates.indexKeyMapping.get(Ints.asList(inner_index)),
                            (int) Math.pow(-1, l)
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
        int[][] hessianIndexKeyMapping = new int[numberOfVariables][numberOfVariables];
        int hessianEntryIndex = 0;
        for (int i = 0; i < numberOfVariables; i++) {
            for (int j = i; j < numberOfVariables; j++) {
                hessianRowIndexes[hessianEntryIndex] = i;
                hessianColumnIndexes[hessianEntryIndex] = j;
                hessianIndexKeyMapping[i][j] = hessianEntryIndex++;
            }
        }

        // Instantiate the chosen objective function class
        objectiveFunction = objectiveFunctionType.build(errorRates, hessianIndexKeyMapping);
    }

    @Override
    public double[] solve() {
        return new double[0];
    }
}
