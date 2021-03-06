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
public class CoordinateDescentSolverTest {
    @Test
    public void testCycleMethodSolver() {
        System.out.println("Rosenbrock Function Coordinate Descent (Cycle Method):\n");
        CoordinateDescentSolver coordinateDescentSolver =
                new CoordinateDescentSolver.Builder(new RosenbrockFunction(), Vectors.dense(-1.2, 1))
                        .method(CoordinateDescentSolver.Method.CYCLE)
                        .maximumNumberOfIterations(10000)
                        .build();
        double[] actualResult = coordinateDescentSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 0.3);

        System.out.println("Quadratic Function Coordinate Descent (Cycle Method):\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = Vectors.dense(new double[]{1, 2});
        coordinateDescentSolver =
                new CoordinateDescentSolver.Builder(new QuadraticFunction(A, b), Vectors.dense(0.0, 0.0))
                        .method(CoordinateDescentSolver.Method.CYCLE)
                        .build();
        actualResult = coordinateDescentSolver.solve().getDenseArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testBackAndForthMethodSolver() {
        System.out.println("Rosenbrock Function Coordinate Descent (Back and Forth Method):\n");
        CoordinateDescentSolver coordinateDescentSolver =
                new CoordinateDescentSolver.Builder(new RosenbrockFunction(), Vectors.dense(-1.2, 1))
                        .method(CoordinateDescentSolver.Method.BACK_AND_FORTH)
                        .maximumNumberOfIterations(10000)
                        .build();
        double[] actualResult = coordinateDescentSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 0.3);

        System.out.println("Quadratic Function Coordinate Descent (Back and Forth Method):\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = Vectors.dense(new double[]{1, 2});
        coordinateDescentSolver =
                new CoordinateDescentSolver.Builder(new QuadraticFunction(A, b), Vectors.dense(0.0, 0.0))
                        .method(CoordinateDescentSolver.Method.BACK_AND_FORTH)
                        .build();
        actualResult = coordinateDescentSolver.solve().getDenseArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testCycleAndJoinEndpointsMethodSolver() {
        System.out.println("Rosenbrock Function Coordinate Descent (Cycle and Join Endpoints Method):\n");
        CoordinateDescentSolver coordinateDescentSolver =
                new CoordinateDescentSolver.Builder(new RosenbrockFunction(), Vectors.dense(-1.2, 1))
                        .method(CoordinateDescentSolver.Method.CYCLE_AND_JOIN_ENDPOINTS)
                        .build();
        double[] actualResult = coordinateDescentSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 0.2);

        System.out.println("Quadratic Function Coordinate Descent (Cycle and Join Endpoints Method):\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = Vectors.dense(new double[]{1, 2});
        coordinateDescentSolver =
                new CoordinateDescentSolver.Builder(new QuadraticFunction(A, b), Vectors.dense(0.0, 0.0))
                        .method(CoordinateDescentSolver.Method.CYCLE_AND_JOIN_ENDPOINTS)
                        .build();
        actualResult = coordinateDescentSolver.solve().getDenseArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
