package org.platanios.learn.optimization;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum NonlinearConjugateGradientMethod {
    FLETCHER_RIEVES,
    POLAK_RIBIERE,
    POLAK_RIBIERE_PLUS,
    HESTENES_STIEFEL,
    FLETCHER_RIEVES_POLAK_RIBIERE,
    /** Based on Y. Dai and Y. Yuan, A nonlinear conjugate gradient method with a strong global convergence property,
     * SIAM Journal on Optimization, 10 (1999), pp. 177–182. */
    DAI_YUAN,
    /** Based on W. W. Hager and H. Zhang, A new conjugate gradient method with guaranteed descent and an efficient line
     * search, SIAM Journal on Optimization, 16 (2005), pp. 170–192. */
    HAGER_ZHANG
}
