package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.function.SumFunction;


/**
 * Created by dcard on 4/21/15.
 */
public class ConsensusAlternatingDirectionsMethodOfMultipliersSolverTest {
    @Test
    public void testNewtonSolver() {
        System.out.println("Consensus ADMM Solver:\n");

        // Setup the world's simplest problem
        // C = 1; D = 0; C >> A; A >> B; D >> B
        // 1 - A
        LinearFunction lf_1mA = new LinearFunction(Vectors.dense(new double[]{-1}), 1);
        // A - B
        LinearFunction lf_AmB = new LinearFunction(Vectors.dense(new double[]{1, -1}), 0);
        // 0 - B
        LinearFunction lf_0mB = new LinearFunction(Vectors.dense(new double[]{-1}), 0);

        // Create a sum function that indexes the variables (A=0, B=1)
        SumFunction sumFunction = new SumFunction.Builder(2).addTerm(lf_1mA,
                new int[] { 0 }).addTerm(lf_AmB, new int[] {0,1}).addTerm(lf_0mB, new int[] {1}).build();

        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusADMMSolver =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder( sumFunction,
                        Vectors.dense(new double[]{0.5,0.5})).build();

        Vector result = consensusADMMSolver.solve();
        System.out.println(result.get(0));   // Should be 1.0
        System.out.println(result.get(1));   // Should be 1.0
        System.out.println('\n');

        /*
        double[] actualResult = newtonRaphsonSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Newton Solver:\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = Vectors.dense(new double[]{1, 2});
        newtonRaphsonSolver = new NewtonSolver.Builder(new QuadraticFunction(A, b),
                Vectors.dense(new double[]{0, 0})).build();
        actualResult = newtonRaphsonSolver.solve().getDenseArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
        */
    }
}
