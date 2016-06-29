package makina.optimization;

import makina.optimization.constraint.LinearInequalityConstraint;
import org.junit.Assert;
import org.junit.Test;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NonlinearInteriorPointSolverTest {
    @Test
    public void testRosenbrockObjective() {
        Vector a1 = Vectors.dense(1.0, 0.0);
        Vector a2 = Vectors.dense(0.0, 1.0);
        NonlinearInteriorPointSolver newtonSolver =
                new NonlinearInteriorPointSolver.Builder(new RosenbrockFunction(), Vectors.dense(-1.2, 1))
                        .addInequalityConstraint(new LinearInequalityConstraint(a1, 1.0))
                        .addInequalityConstraint(new LinearInequalityConstraint(a1.mult(-1), 1.0))
                        .addInequalityConstraint(new LinearInequalityConstraint(a2, 1.0))
                        .addInequalityConstraint(new LinearInequalityConstraint(a2.mult(-1), 1.0))
                        .loggingLevel(5)
                        .build();
        double[] actualResult = newtonSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
