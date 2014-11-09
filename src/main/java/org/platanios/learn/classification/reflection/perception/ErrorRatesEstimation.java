package org.platanios.learn.classification.reflection.perception;

/**
 * Handles the set up of the numerical optimization problem that needs to be solved in order to estimate error rates of
 * several approximations to a single function, by using only the agreement rates of those functions on an unlabeled set
 * of data.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimation {
    /** Data structure containing all the data needed for estimating the error rates. */
    private final EstimationData data;
    /** The wrapper of the solver used for solving the numerical optimization problem involved in the estimation of the
     * error rates. */
    private final OptimizationProblem optimizationProblem;

    /**
     * A class used to build {@link ErrorRatesEstimation} objects.
     */
    public static class Builder {
        /** A data structure ({@link EstimationData}) which contains  all the data required to setup and solve the
         * numerical optimization problem that needs to be solved for estimating the error rates. */
        private final EstimationData data;
        /** The type of objective function to minimize (e.g. minimize dependency, scaled dependency, etc.). */
        private ObjectiveFunctionType objectiveFunctionType = ObjectiveFunctionType.DEPENDENCY;
        /** The type of optimization problem solver to use. */
        private OptimizationSolverType optimizationSolverType = OptimizationSolverType.IP_OPT;

        /**
         * Constructs a builder object for the {@link ErrorRatesEstimation} class.
         *
         * @param   data    A data structure ({@link EstimationData}) which contains  all the data required to setup and
         *                  solve the numerical optimization problem that needs to be solved for estimating the error
         *                  rates.
         */
        public Builder(EstimationData data) {
            this.data = data;
        }

        /**
         * Sets the type of the objective function to be minimized.
         *
         * @param   objectiveFunctionType   The type of objective function to minimize (e.g. minimize dependency, scaled
         *                                  dependency, etc.).
         * @return                          The current builder object after setting this property.
         */
        public Builder objectiveFunctionType(ObjectiveFunctionType objectiveFunctionType) {
            this.objectiveFunctionType = objectiveFunctionType;
            return this;
        }

        /**
         * Sets the type of optimization solver to use.
         *
         * @param   optimizationSolverType  The type of optimization solver to use.
         * @return                          The current builder object after setting this property.
         */
        public Builder optimizationSolverType(OptimizationSolverType optimizationSolverType) {
            this.optimizationSolverType = optimizationSolverType;
            return this;
        }

        /**
         * Builds a {@link ErrorRatesEstimation} object using the current builder object.
         *
         * @return  A {@link ErrorRatesEstimation} object built using the current builder object.
         */
        public ErrorRatesEstimation build() {
            return new ErrorRatesEstimation(this);
        }
    }

    /**
     * Sets up the optimization problem that needs to be solved for estimating the error rates, using the settings
     * included in the provided {@link ErrorRatesEstimation.Builder} object, and also initializes the numerical
     * optimization solver that is used to solve it.
     *
     * @param   builder The builder object containing the settings to be used.
     */
    private ErrorRatesEstimation(Builder builder) {
        this.data = builder.data;
        this.optimizationProblem = OptimizationProblems.build(data.getNumberOfFunctions(),
                                                              data.getHighestOrder(),
                                                              data.getErrorRates(),
                                                              data.getAgreementRates(),
                                                              builder.objectiveFunctionType,
                                                              builder.optimizationSolverType);
    }

    /**
     * Solves the numerical optimization problem and returns the obtained error rates estimates in an
     * {@link EstimationData} structure.
     *
     * @return  The obtained error rate estimates in an {@link EstimationData} structure.
     */
    public EstimationData solve() {
        return data.setErrorRatesValues(optimizationProblem.solve());
    }
}
