package makina.optimization;

import makina.optimization.function.AbstractFunction;
import makina.optimization.linesearch.StrongWolfeInterpolationLineSearch;
import org.junit.Assert;
import org.junit.Test;
import makina.math.matrix.Vectors;
import makina.optimization.linesearch.StepSizeInitializationMethod;

/**
 * @author Emmanouil Antonios Platanios
 */
public class QuasiNewtonSolverTest {
    @Test
    public void testQuasiNewtonDFPSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton DFP Solver:\n");
        QuasiNewtonSolver quasiNewtonDFPSolver = new QuasiNewtonSolver.Builder(new RosenbrockFunction(),
                                                                               Vectors.dense(-1.2, 1))
                .method(QuasiNewtonSolver.Method.DAVIDON_FLETCHER_POWELL)
                .build();
        double[] actualResult = quasiNewtonDFPSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
    
    @Test
    public void testQuasiNewtonBFGSSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton BFGS Solver:\n");
        QuasiNewtonSolver quasiNewtonBFGSSolver = new QuasiNewtonSolver.Builder(new RosenbrockFunction(),
                                                                                Vectors.dense(-1.2, 1))
                .method(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                .lineSearch(new makina.optimization.linesearch.NoLineSearch(1.0))
                .loggingLevel(5)
                .build();
        double[] actualResult = quasiNewtonBFGSSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

//    @Test
//    public void testQuasiNewtonSR1Solver() {
//        System.out.println("Rosenbrock Function Quasi-Newton SR1 Solver:\n");
//        QuasiNewtonSolver quasiNewtonSR1Solver =
//                new QuasiNewtonSolver.Builder(new RosenbrockFunction(),
//                                              VectorFactory.dense(new double[] { -1.2, 1 }))
//                .method(QuasiNewtonSolver.Method.SYMMETRIC_RANK_ONE)
//                .build();
//        double[] actualResult = quasiNewtonSR1Solver.solve().getDenseArray();
//        double[] expectedResult = new double[] { 1, 1 };
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
//    }
//
//    @Test
//    public void testQuasiNewtonBroydenSolver() {
//        System.out.println("Rosenbrock Function Quasi-Newton Broyden Solver:\n");
//        QuasiNewtonSolver quasiNewtonBroydenSolver =
//                new QuasiNewtonSolver.Builder(new RosenbrockFunction(),
//                                              VectorFactory.dense(new double[] { -1.2, 1 }))
//                        .method(QuasiNewtonSolver.Method.BROYDEN)
//                        .build();
//        double[] actualResult = quasiNewtonBroydenSolver.solve().getDenseArray();
//        double[] expectedResult = new double[] { 1, 1 };
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
//    }
    
    @Test
    public void testQuasiNewtonLBFGSSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton L-BFGS Solver:\n");
        AbstractFunction objective = new RosenbrockFunction();
        StrongWolfeInterpolationLineSearch lineSearch = new StrongWolfeInterpolationLineSearch(objective, 1e-4, 0.9, 10.0);
        lineSearch.setStepSizeInitializationMethod(StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE);
        QuasiNewtonSolver quasiNewtonLBFGSSolver = new QuasiNewtonSolver.Builder(objective, Vectors.dense(-1.2, 1))
                .method(QuasiNewtonSolver.Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                .lineSearch(lineSearch)
                .loggingLevel(5)
                .m(10)
                .build();
        double[] actualResult = quasiNewtonLBFGSSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
