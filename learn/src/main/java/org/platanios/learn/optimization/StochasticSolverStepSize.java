package org.platanios.learn.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum StochasticSolverStepSize {
    CONSTANT {
        @Override
        public double compute(int iteration, double... parameters) {
            if (parameters.length < 1)
                throw new IllegalArgumentException("The parameters array must have length 1 (it must hold the " +
                                                           "constant step size value).");
            if (parameters.length > 1)
                logger.info("Only one parameter is needed for the compute() method. The rest will be ignored.");
            return parameters[0];
        }
    },
    SCALED {
        @Override
        public double compute(int iteration, double... parameters) {
            if (parameters.length < 2)
                throw new IllegalArgumentException("The parameters array must have length 2 (it must hold the " +
                                                           "tau and the kappa parameter).");
            if (parameters[0] < 0)
                throw new IllegalArgumentException("The value of the tau parameter (i.e. parameters[0]) must be >= 0.");
            if (parameters[1] <= 0.5 || parameters[1] > 1)
                throw new IllegalArgumentException("The value of the kappa parameter (i.e. parameters[1]) " +
                                                           "must be in the interval (0.5,1].");
            if (parameters.length > 2)
                logger.info("Only one parameter is needed for the compute method. The rest will be ignored.");
            return Math.pow(parameters[0] + iteration + 1, -parameters[1]);
        }
    };

    private static final Logger logger = LogManager.getLogger("Stochastic Optimization / Step Size");

    public abstract double compute(int iteration, double... parameters);
}