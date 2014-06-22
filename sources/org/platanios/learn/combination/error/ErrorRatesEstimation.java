package org.platanios.learn.combination.error;

/**
 * Handles the set up of the numerical optimization problem that needs to be solved in order to estimate error rates of
 * several approximations to a single function, by using only the agreement rates of those functions on an unlabeled set
 * of data.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimation {
    /** Data structure containing all the data needed for estimating the error rates. */
    private EstimationData data;
    /** The wrapper of the solver used for solving the numerical optimization problem involved in the estimation of the
     * error rates. */
    private KnitroOptimizationProblem optimizationProblem;

    /**
     * Sets up the optimization problem that needs to be solved for estimating the error rates and initializes the
     * numerical optimization solver that is used to solve it. That solver is currently the KNITRO solver.
     *
     * @param   data    A data structure ({@link org.platanios.learn.combination.error.EstimationData}) which contains
     *                  all the data required to setup and solve the numerical optimization problem.
     */
    public ErrorRatesEstimation(EstimationData data) {
        this.data = data;
        this.optimizationProblem = new KnitroOptimizationProblem(data.getNumberOfFunctions(),
                                                                 data.getMaximumOrder(),
                                                                 data.getErrorRates(),
                                                                 data.getAgreementRates());
    }

    /**
     * Solves the numerical optimization problem and returns the obtained error rates estimates in an
     * {@link org.platanios.learn.combination.error.EstimationData} structure.
     *
     * @return  The obtained error rate estimates in an
     *          {@link org.platanios.learn.combination.error.EstimationData} structure.
     */
    public EstimationData solve() {
        data.setErrorRatesValues(optimizationProblem.solve());
        return data;
    }
}
