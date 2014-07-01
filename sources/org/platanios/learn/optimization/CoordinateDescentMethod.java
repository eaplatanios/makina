package org.platanios.learn.optimization;

/**
 * An enumeration of all currently supported coordinate descent methods.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum CoordinateDescentMethod {
    /** The algorithm cycles over the coordinates (after it uses the last coordinate it goes back to the first one). */
    CYCLE,
    /** The algorithm goes back and forth over the coordinates (it uses the coordinates in the following order: 1, 2,
     * ..., n-1, n, n-1, ..., 2, 1, 2, ...). */
    BACK_AND_FORTH,
    /** The algorithm cycles over the coordinates as with the {@link #CYCLE} method, but after each cycle completes, it
     * takes a step in the direction computed as the difference between the first point in the cycle and the last point
     * in the cycle. */
    CYCLE_AND_JOIN_ENDPOINTS
}
