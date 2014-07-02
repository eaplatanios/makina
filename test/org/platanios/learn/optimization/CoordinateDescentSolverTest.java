package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CoordinateDescentSolverTest {
    @Test
    public void testCycleMethodSolver() {
        System.out.println("Rosenbrock Function Coordinate Descent (Cycle Method):\n");
        CoordinateDescentSolver coordinateDescentSolver = new CoordinateDescentSolver(new RosenbrockFunction(),
                                                                                      new double[] { -1.2, 1 });
        coordinateDescentSolver.setMethod(CoordinateDescentSolver.Method.CYCLE);
        double[] actualResult = coordinateDescentSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 0.2);

        System.out.println("Quadratic Function Coordinate Descent (Cycle Method):\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 1, 2 });
        coordinateDescentSolver = new CoordinateDescentSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        coordinateDescentSolver.setMethod(CoordinateDescentSolver.Method.CYCLE);
        actualResult = coordinateDescentSolver.solve().toArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testBackAndForthMethodSolver() {
        System.out.println("Rosenbrock Function Coordinate Descent (Back and Forth Method):\n");
        CoordinateDescentSolver coordinateDescentSolver = new CoordinateDescentSolver(new RosenbrockFunction(),
                                                                                      new double[] { -1.2, 1 });
        coordinateDescentSolver.setMethod(CoordinateDescentSolver.Method.BACK_AND_FORTH);
        double[] actualResult = coordinateDescentSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 0.2);

        System.out.println("Quadratic Function Coordinate Descent (Back and Forth Method):\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 1, 2 });
        coordinateDescentSolver = new CoordinateDescentSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        coordinateDescentSolver.setMethod(CoordinateDescentSolver.Method.BACK_AND_FORTH);
        actualResult = coordinateDescentSolver.solve().toArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testCycleAndJoinEndpointsMethodSolver() {
        System.out.println("Rosenbrock Function Coordinate Descent (Cycle and Join Endpoints Method):\n");
        CoordinateDescentSolver coordinateDescentSolver = new CoordinateDescentSolver(new RosenbrockFunction(),
                                                                                      new double[] { -1.2, 1 });
        coordinateDescentSolver.setMethod(CoordinateDescentSolver.Method.CYCLE_AND_JOIN_ENDPOINTS);
        double[] actualResult = coordinateDescentSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 0.2);

        System.out.println("Quadratic Function Coordinate Descent (Cycle and Join Endpoints Method):\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 1, 2 });
        coordinateDescentSolver = new CoordinateDescentSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        coordinateDescentSolver.setMethod(CoordinateDescentSolver.Method.CYCLE_AND_JOIN_ENDPOINTS);
        actualResult = coordinateDescentSolver.solve().toArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
