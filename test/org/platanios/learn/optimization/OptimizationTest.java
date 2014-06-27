package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.optimization.function.Function;
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
        NewtonRaphsonSolver newtonRaphsonSolver = new NewtonRaphsonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        double[] actualResult = newtonRaphsonSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Newton's Method:\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 1 }, { -3, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 6, 2 });
        newtonRaphsonSolver = new NewtonRaphsonSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
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
        System.out.println("Rosenbrock Function Fletcher-Reeves Solver:\n");
        NonlinearConjugateGradientSolver fletcherReevesSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(),
                                                     new double[] { -1.2, 1 },
                                                     NonlinearConjugateGradientMethod.FLETCHER_RIEVES);
        double[] actualResult = fletcherReevesSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testPolakRibiereSolver() {
        System.out.println("Rosenbrock Function Polak-Ribiere Solver:\n");
        NonlinearConjugateGradientSolver polakRibiereSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(),
                                                     new double[] { -1.2, 1 },
                                                     NonlinearConjugateGradientMethod.POLAK_RIBIERE);
        double[] actualResult = polakRibiereSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testPolakRibierePlusSolver() {
        System.out.println("Rosenbrock Function Polak-Ribiere+ Solver:\n");
        NonlinearConjugateGradientSolver polakRibierePlusSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(),
                                                     new double[] { -1.2, 1 },
                                                     NonlinearConjugateGradientMethod.POLAK_RIBIERE_PLUS);
        double[] actualResult = polakRibierePlusSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testHestenesStiefelSolver() {
        System.out.println("Rosenbrock Function Hestenes-Stiefel Solver:\n");
        NonlinearConjugateGradientSolver hestenesStiefelSolver = new
                NonlinearConjugateGradientSolver(new RosenbrockFunction(),
                                                 new double[] { -1.2, 1 },
                                                 NonlinearConjugateGradientMethod.HESTENES_STIEFEL);
        double[] actualResult = hestenesStiefelSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    class RosenbrockFunction implements Function {
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
