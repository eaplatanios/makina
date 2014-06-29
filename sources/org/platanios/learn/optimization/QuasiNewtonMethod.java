package org.platanios.learn.optimization;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum QuasiNewtonMethod {
    /** The Davidon–Fletcher–Powell algorithm. This algorithm is less effective than BROYDEN_FLETCHER_GOLDFARB_SHANNO at correcting bad Hessian
     * approximations. Both this method and the BROYDEN_FLETCHER_GOLDFARB_SHANNO method preserve the positive-definiteness of the Hessian matrix. */
    DAVIDON_FLETCHER_POWELL,
    /** The Broyden–Fletcher–Goldfarb–Shanno algorithm. This algorithm is very good at correcting bad Hessian
     * approximations. */
    BROYDEN_FLETCHER_GOLDFARB_SHANNO,
    LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO,
    /** The Symmetric-Rank-1 algorithm. This method may produce indefinite Hessian approximations. Furthermore, the
     * basic SYMMETRIC_RANK_ONE method may break down and that is why here it has been implemented with a skipping method to help
     * prevent such cases. */
    SYMMETRIC_RANK_ONE,
    BROYDEN

}
