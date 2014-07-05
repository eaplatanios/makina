package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class QuasiNewtonSolverTest {
    @Test
    public void testQuasiNewtonDFPSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton DFP Solver:\n");
        QuasiNewtonSolver quasiNewtonDFPSolver =
                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        quasiNewtonDFPSolver.setMethod(QuasiNewtonSolver.Method.DAVIDON_FLETCHER_POWELL);
        double[] actualResult = quasiNewtonDFPSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testQuasiNewtonBFGSSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton BFGS Solver:\n");
        QuasiNewtonSolver quasiNewtonBFGSSolver =
                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        quasiNewtonBFGSSolver.setMethod(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO);
        double[] actualResult = quasiNewtonBFGSSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

//    @Test
//    public void testQuasiNewtonSR1Solver() {
//        System.out.println("Rosenbrock Function Quasi-Newton SR1 Solver:\n");
//        QuasiNewtonSolver quasiNewtonSR1Solver =
//                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
//        quasiNewtonSR1Solver.setMethod(QuasiNewtonSolver.Method.SYMMETRIC_RANK_ONE);
//        double[] actualResult = quasiNewtonSR1Solver.solve().getArray();
//        double[] expectedResult = new double[] { 1, 1 };
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
//    }
//
//    @Test
//    public void testQuasiNewtonBroydenSolver() {
//        System.out.println("Rosenbrock Function Quasi-Newton Broyden Solver:\n");
//        QuasiNewtonSolver quasiNewtonBroydenSolver =
//                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
//        quasiNewtonBroydenSolver.setMethod(QuasiNewtonSolver.Method.BROYDEN);
//        double[] actualResult = quasiNewtonBroydenSolver.solve().getArray();
//        double[] expectedResult = new double[] { 1, 1 };
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
//    }

    @Test
    public void testQuasiNewtonLBFGSSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton L-BFGS Solver:\n");
        QuasiNewtonSolver quasiNewtonLBFGSSolver =
                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        quasiNewtonLBFGSSolver.setMethod(QuasiNewtonSolver.Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO);
        double[] actualResult = quasiNewtonLBFGSSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
