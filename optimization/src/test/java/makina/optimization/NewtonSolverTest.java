package makina.optimization;

import org.junit.Assert;
import org.junit.Test;
import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.optimization.constraint.LinearEqualityConstraint;
import makina.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NewtonSolverTest {
    @Test
    public void testNewtonSolver() {
        System.out.println("Rosenbrock Function Newton Solver:\n");
        NewtonSolver newtonSolver =
                new NewtonSolver.Builder(new RosenbrockFunction(), Vectors.dense(-1.2, 1)).build();
        double[] actualResult = newtonSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Newton Solver:\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = Vectors.dense(new double[]{1, 2});
        newtonSolver = new NewtonSolver.Builder(new QuadraticFunction(A, b),
                                                       Vectors.dense(new double[]{0, 0})).build();
        actualResult = newtonSolver.solve().getDenseArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testSimpleFunctionNewtonSolver() {
        NewtonSolver newtonSolver = new NewtonSolver.Builder(new SimpleFunction(),
                                                             Vectors.dense(new double[]{0,0})).build();
        double[] actualResult = newtonSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 2, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testLinearEqualityConstrainedSimpleFunctionNewtonSolver() {
        NewtonSolver newtonSolver = new NewtonSolver.Builder(new SimpleFunction(),
                                                             Vectors.dense(new double[]{0, 0}))
                .addLinearEqualityConstraint(new LinearEqualityConstraint(Vectors.dense(new double[] {1.0, 4.0}), 3))
                .loggingLevel(5)
                .build();
        double[] actualResult = newtonSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 5.0 / 3.0, 1.0 / 3.0 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
