package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Solver {
    public void updateDirection();
    public void updateStepSize();
    public void updatePoint();
    public boolean checkForConvergence();
    public RealVector solve();
}
