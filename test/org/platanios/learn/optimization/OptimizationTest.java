package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class OptimizationTest {
    @Test
    public void testSteepestDescentSolver() {
        System.out.println("Rosenbrock Function Gradient Descent:\n");
        GradientDescentSolver gradientDescentSolver = new GradientDescentSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        double[] actualResult = gradientDescentSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Gradient Descent:\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 2 }, { 2, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 4, 2 });
        gradientDescentSolver = new GradientDescentSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        actualResult = gradientDescentSolver.solve().toArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testNewtonsMethodSolver() {
        System.out.println("Rosenbrock Function Newton's Method:\n");
        NewtonSolver newtonRaphsonSolver = new NewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        double[] actualResult = newtonRaphsonSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Newton's Method:\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 1 }, { -3, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 6, 2 });
        newtonRaphsonSolver = new NewtonSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        actualResult = newtonRaphsonSolver.solve().toArray();
        expectedResult = new double[] { 1, 5 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testConjugateGradientSolver() {
        System.out.println("Quadratic Function Conjugate Gradient:\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 2 }, { 2, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 4, 2 });
        ConjugateGradientSolver conjugateGradientSolver = new ConjugateGradientSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        double[] actualResult = conjugateGradientSolver.solve().toArray();
        double[] expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testFletcherReevesSolver() {
        System.out.println("Rosenbrock Function Fletcher-Reeves No-Restart Solver:\n");
        NonlinearConjugateGradientSolver fletcherReevesSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherReevesSolver.setMethod(NonlinearConjugateGradientMethod.FLETCHER_RIEVES);
        fletcherReevesSolver.setRestartMethod(NonlinearConjugateGradientRestartMethod.NO_RESTART);
        double[] actualResult = fletcherReevesSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Fletcher-Reeves Gradients-Orthogonality-Check-Restart Solver:\n");
        fletcherReevesSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherReevesSolver.setMethod(NonlinearConjugateGradientMethod.FLETCHER_RIEVES);
        fletcherReevesSolver.setRestartMethod(NonlinearConjugateGradientRestartMethod.GRADIENTS_ORTHOGONALITY_CHECK);
        actualResult = fletcherReevesSolver.solve().toArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testPolakRibiereSolver() {
        System.out.println("Rosenbrock Function Polak-Ribiere Solver:\n");
        NonlinearConjugateGradientSolver polakRibiereSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        polakRibiereSolver.setMethod(NonlinearConjugateGradientMethod.POLAK_RIBIERE);
        double[] actualResult = polakRibiereSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testPolakRibierePlusSolver() {
        System.out.println("Rosenbrock Function Polak-Ribiere+ Solver:\n");
        NonlinearConjugateGradientSolver polakRibierePlusSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        polakRibierePlusSolver.setMethod(NonlinearConjugateGradientMethod.POLAK_RIBIERE_PLUS);
        double[] actualResult = polakRibierePlusSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testHestenesStiefelSolver() {
        System.out.println("Rosenbrock Function Hestenes-Stiefel Solver:\n");
        NonlinearConjugateGradientSolver hestenesStiefelSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        hestenesStiefelSolver.setMethod(NonlinearConjugateGradientMethod.HESTENES_STIEFEL);
        double[] actualResult = hestenesStiefelSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testFletcherRievesPolakRibiereSolver() {
        System.out.println("Rosenbrock Function Fletcher-Rieves-Polak-Ribiere Solver:\n");
        NonlinearConjugateGradientSolver fletcherRievesPolakRibiereSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherRievesPolakRibiereSolver.setMethod(NonlinearConjugateGradientMethod.FLETCHER_RIEVES_POLAK_RIBIERE);
        double[] actualResult = fletcherRievesPolakRibiereSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testDaiYuanSolver() {
        System.out.println("Rosenbrock Function Dai-Yuan Solver:\n");
        NonlinearConjugateGradientSolver daiYuanSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        daiYuanSolver.setMethod(NonlinearConjugateGradientMethod.DAI_YUAN);
        double[] actualResult = daiYuanSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testHagerZhangSolver() {
        System.out.println("Rosenbrock Function Hager-Zhang Solver:\n");
        NonlinearConjugateGradientSolver hagerZhangSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        hagerZhangSolver.setMethod(NonlinearConjugateGradientMethod.HAGER_ZHANG);
        double[] actualResult = hagerZhangSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testQuasiNewtonDFPSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton DFP Solver:\n");
        QuasiNewtonSolver quasiNewtonDFPSolver =
                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        quasiNewtonDFPSolver.setMethod(QuasiNewtonMethod.DAVIDON_FLETCHER_POWELL);
        double[] actualResult = quasiNewtonDFPSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testQuasiNewtonBFGSSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton BFGS Solver:\n");
        QuasiNewtonSolver quasiNewtonBFGSSolver =
                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        quasiNewtonBFGSSolver.setMethod(QuasiNewtonMethod.BROYDEN_FLETCHER_GOLDFARB_SHANNO);
        double[] actualResult = quasiNewtonBFGSSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testQuasiNewtonSR1Solver() {
        System.out.println("Rosenbrock Function Quasi-Newton SR1 Solver:\n");
        QuasiNewtonSolver quasiNewtonSR1Solver =
                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        quasiNewtonSR1Solver.setMethod(QuasiNewtonMethod.SYMMETRIC_RANK_ONE);
        double[] actualResult = quasiNewtonSR1Solver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testQuasiNewtonBroydenSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton Broyden Solver:\n");
        QuasiNewtonSolver quasiNewtonBroydenSolver =
                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        quasiNewtonBroydenSolver.setMethod(QuasiNewtonMethod.BROYDEN);
        double[] actualResult = quasiNewtonBroydenSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testQuasiNewtonLBFGSSolver() {
        System.out.println("Rosenbrock Function Quasi-Newton L-BFGS Solver:\n");
        QuasiNewtonSolver quasiNewtonLBFGSSolver =
                new QuasiNewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        quasiNewtonLBFGSSolver.setMethod(QuasiNewtonMethod.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO);
        double[] actualResult = quasiNewtonLBFGSSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    class RosenbrockFunction extends AbstractFunction {
        public double computeValue(RealVector optimizationVariables) {
            double x1 = optimizationVariables.getEntry(0);
            double x2 = optimizationVariables.getEntry(1);
            return 100 * Math.pow(x2 - Math.pow(x1, 2), 2) + Math.pow(1 - x1, 2);
        }

        public RealVector computeGradient(RealVector optimizationVariables) {
            double x1 = optimizationVariables.getEntry(0);
            double x2 = optimizationVariables.getEntry(1);
            double dx1 = - 400 * (x2 - Math.pow(x1, 2)) * x1 - 2 * (1 - x1);
            double dx2 = 200 * (x2 - Math.pow(x1, 2));
            return new ArrayRealVector(new double[] { dx1, dx2 });
        }

        public RealMatrix computeHessian(RealVector optimizationVariables) {
            double x1 = optimizationVariables.getEntry(0);
            double x2 = optimizationVariables.getEntry(1);
            double dx1x1 = 1200 * Math.pow(x1, 2) - 400 * x2 + 2;
            double dx1x2 = - 400 * x1;
            double dx2x2 = 200;
            return new Array2DRowRealMatrix(new double[][] { { dx1x1, dx1x2 }, { dx1x2, dx2x2 } });
        }
    }
}
