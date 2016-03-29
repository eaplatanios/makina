package org.platanios.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.math.matrix.Vector;

import java.text.DecimalFormat;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Solver {
    DecimalFormat DECIMAL_FORMAT = new DecimalFormat("0.0000000000E0");
    Logger logger = LogManager.getFormatterLogger("Optimization");

    Vector solve();
}
