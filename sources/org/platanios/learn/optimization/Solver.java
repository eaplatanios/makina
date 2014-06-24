package org.platanios.learn.optimization;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Solver {
    public void updateDirection();
    public void updatePoint();
    public boolean checkForConvergence();
    public double[] solve();
}
