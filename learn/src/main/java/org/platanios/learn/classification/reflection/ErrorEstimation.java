package org.platanios.learn.classification.reflection;

/**
 * Handles the set up of the numerical optimization problem that needs to be solved in order to estimate error rates of
 * several approximations to a single function, by using only the agreement rates of those functions on an unlabeled set
 * of data.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimation {
    /** Data structure containing all the data needed for estimating the error rates. */
    private final ErrorEstimationData data;
    /** The wrapper of the solver used for solving the numerical optimization problem involved in the estimation of the
     * error rates. */
    private final ErrorEstimationOptimization optimizationProblem;

    /**
     * A class used to build {@link ErrorEstimation} objects.
     */
    public static class Builder {
        /** A data structure ({@link ErrorEstimationData}) which contains  all the data required to setup and solve the
         * numerical optimization problem that needs to be solved for estimating the error rates. */
        private final ErrorEstimationData data;
        /** The type of objective function to minimize (e.g. minimize dependency, scaled dependency, etc.). */
        private ErrorEstimationObjective objectiveFunctionType = ErrorEstimationObjective.DEPENDENCY;
        /** The type of optimization problem solver to use. */
        private ErrorEstimationInternalSolver internalSolver = ErrorEstimationInternalSolver.IP_OPT;

        /**
         * Constructs a builder object for the {@link ErrorEstimation} class.
         *
         * @param   data    A data structure ({@link ErrorEstimationData}) which contains  all the data required to
         *                  setup and solve the numerical optimization problem that needs to be solved for estimating
         *                  the error rates.
         */
        public Builder(ErrorEstimationData data) {
            this.data = data;
        }

        /**
         * Sets the type of the objective function to be minimized.
         *
         * @param   objectiveFunctionType   The type of objective function to minimize (e.g. minimize dependency, scaled
         *                                  dependency, etc.).
         * @return                          The current builder object after setting this property.
         */
        public Builder objectiveFunctionType(ErrorEstimationObjective objectiveFunctionType) {
            this.objectiveFunctionType = objectiveFunctionType;
            return this;
        }

        /**
         * Sets the type of optimization solver to use.
         *
         * @param   optimizationSolverType  The type of optimization solver to use.
         * @return                          The current builder object after setting this property.
         */
        public Builder optimizationSolverType(ErrorEstimationInternalSolver optimizationSolverType) {
            this.internalSolver = optimizationSolverType;
            return this;
        }

        /**
         * Builds a {@link ErrorEstimation} object using the current builder object.
         *
         * @return  A {@link ErrorEstimation} object built using the current builder object.
         */
        public ErrorEstimation build() {
            return new ErrorEstimation(this);
        }
    }

    /**
     * Sets up the optimization problem that needs to be solved for estimating the error rates, using the settings
     * included in the provided {@link ErrorEstimation.Builder} object, and also initializes the numerical
     * optimization solver that is used to solve it.
     *
     * @param   builder The builder object containing the settings to be used.
     */
    private ErrorEstimation(Builder builder) {
        data = builder.data;
        optimizationProblem = builder.internalSolver.buildOptimizationProblem(data.getNumberOfFunctions(),
                                                                              data.getHighestOrder(),
                                                                              data.getErrorRates(),
                                                                              data.getAgreementRates(),
                                                                              builder.objectiveFunctionType);
    }

    /**
     * Solves the numerical optimization problem and returns the obtained error rates estimates in an
     * {@link ErrorEstimationData} structure.
     *
     * @return  The obtained error rate estimates in an {@link ErrorEstimationData} structure.
     */
    public ErrorEstimationData solve() {
        return data.setErrorRatesValues(optimizationProblem.solve());
    }
}