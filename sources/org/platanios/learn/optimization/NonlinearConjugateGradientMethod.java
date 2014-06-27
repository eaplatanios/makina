package org.platanios.learn.optimization;

/**
 * An enumeration of the currently supported nonlinear conjugate gradient optimization methods. They are very similar to
 * each other, but there are some significant differences that one should consider when choosing which method to use.
 * Here we provide a short description of some of those differences.
 * <br><br>
 * One weakness of the Fletcher-Rieves method is that when the algorithm generates a bad direction and a tiny step size,
 * then the next direction and the next step are also likely to be poor. The Polak-Ribiere method effectively performs a
 * restart when that happens. The same holds for the Polak-Ribiere+ and the Hestenes-Stiefel methods. The
 * Fletcher-Rieves-Polak-Ribiere method also deals well with that issue and at the same time retains the global
 * convergence properties of the Fletcher-Rieves method. The Fletcher-Rieves method, if used instead of one of the
 * other methods, it should be used along with some restart strategy in order to avoid that problem.
 * <br><br>
 * The Polak-Ribiere method can end up cycling infinitely without converging to a solution. In that case it might be
 * better to use the Polak-Ribiere+ method. Furthermore, we can prove global convergence results for the
 * Fletcher-Rieves, the Fletcher-Rieves-Polak-Ribiere, the Dai-Yuan and the Hager-Zhang methods, but we cannot prove
 * global convergence results for the Polak-Ribiere method. However, in practice the Polak-Ribiere+ method seems to be
 * the fastest one (and we can prove global convergence results for this method when used with certain line search
 * algorithms).
 * <br><br>
 * The Dai-Yuan method is based on the following paper: Y. Dai and Y. Yuan, A nonlinear conjugate gradient method with a
 * strong global convergence property, SIAM Journal on Optimization, 10 (1999), pp. 177&#45;182.
 * <br><br>
 * The Hager-Zhang method is based on the following paper: W. W. Hager and H. Zhang, A new conjugate gradient method
 * with guaranteed descent and an efficient line search, SIAM Journal on Optimization, 16 (2005), pp. 170&#45;192.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum NonlinearConjugateGradientMethod {
    FLETCHER_RIEVES,
    POLAK_RIBIERE,
    POLAK_RIBIERE_PLUS,
    HESTENES_STIEFEL,
    FLETCHER_RIEVES_POLAK_RIBIERE,
    DAI_YUAN,
    HAGER_ZHANG
}
