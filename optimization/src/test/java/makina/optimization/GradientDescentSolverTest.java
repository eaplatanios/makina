package makina.optimization;

import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.optimization.function.QuadraticFunction;
import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientDescentSolverTest {
    @Test
    public void testGradientDescentSolver() {
        System.out.println("Rosenbrock Function Gradient Descent Solver:\n");
        GradientDescentSolver gradientDescentSolver =
                new GradientDescentSolver.Builder(new RosenbrockFunction(), Vectors.dense(-1.2, 1))
                        .maximumNumberOfIterations(5000)
                        .build();
        double[] actualResult = gradientDescentSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Gradient Descent Solver:\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = Vectors.dense(new double[]{1, 2});
        gradientDescentSolver = new GradientDescentSolver.Builder(new QuadraticFunction(A, b),
                                                                  Vectors.dense(0.0, 0.0)).build();
        actualResult = gradientDescentSolver.solve().getDenseArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
