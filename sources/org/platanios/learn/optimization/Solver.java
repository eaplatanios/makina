package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.RealVector;

import java.text.DecimalFormat;

/**
 * @author Emmanouil Antonios Platanios
 */
interface Solver {
    static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.0000000000E0");

    public RealVector solve();
}
