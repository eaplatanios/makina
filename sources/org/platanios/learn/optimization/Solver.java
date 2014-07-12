package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;

import java.text.DecimalFormat;

/**
 * @author Emmanouil Antonios Platanios
 */
interface Solver {
    static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.0000000000E0");

    public Vector solve();
}
