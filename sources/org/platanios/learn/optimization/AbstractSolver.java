package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractSolver implements Solver {
    int currentIteration;
    RealVector currentPoint;

    public RealVector solve() {
        printHeader();
        while (!checkTerminationConditions()) {
            iterationUpdate();
            currentIteration++;
            printIteration();
        }

        printTerminationMessage();

        return currentPoint;
    }
}
