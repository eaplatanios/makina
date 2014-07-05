package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NonlinearConjugateGradientSolverTest {
    @Test
    public void testFletcherReevesSolver() {
        System.out.println("Rosenbrock Function Fletcher-Reeves No-Restart Solver:\n");
        NonlinearConjugateGradientSolver fletcherReevesSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherReevesSolver.setMethod(NonlinearConjugateGradientSolver.Method.FLETCHER_RIEVES);
        fletcherReevesSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.NO_RESTART);
        double[] actualResult = fletcherReevesSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Fletcher-Reeves N-Step-Restart Solver:\n");
        fletcherReevesSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherReevesSolver.setMethod(NonlinearConjugateGradientSolver.Method.FLETCHER_RIEVES);
        fletcherReevesSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.N_STEP);
        actualResult = fletcherReevesSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Fletcher-Reeves Gradients-Orthogonality-Check-Restart Solver:\n");
        fletcherReevesSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherReevesSolver.setMethod(NonlinearConjugateGradientSolver.Method.FLETCHER_RIEVES);
        fletcherReevesSolver.setRestartMethod(
                NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK
        );
        actualResult = fletcherReevesSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testPolakRibiereSolver() {
//        System.out.println("Rosenbrock Function Polak-Ribiere Solver:\n");
//        NonlinearConjugateGradientSolver polakRibiereSolver =
//                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
//        polakRibiereSolver.setMethod(NonlinearConjugateGradientSolver.Method.POLAK_RIBIERE);
//        polakRibiereSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.NO_RESTART);
//        double[] actualResult = polakRibiereSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

//        System.out.println("Rosenbrock Function Polak-Ribiere N-Step-Restart Solver:\n");
//        polakRibiereSolver =
//                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
//        polakRibiereSolver.setMethod(NonlinearConjugateGradientSolver.Method.POLAK_RIBIERE);
//        polakRibiereSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.N_STEP);
//        actualResult = polakRibiereSolver.solve().getArray();
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Polak-Ribiere Gradients-Orthogonality-Check-Restart Solver:\n");
        NonlinearConjugateGradientSolver polakRibiereSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        polakRibiereSolver.setMethod(NonlinearConjugateGradientSolver.Method.POLAK_RIBIERE);
        polakRibiereSolver.setRestartMethod(
                NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK
        );
        double[] actualResult = polakRibiereSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testPolakRibierePlusSolver() {
//        System.out.println("Rosenbrock Function Polak-Ribiere+ Solver:\n");
//        NonlinearConjugateGradientSolver polakRibierePlusSolver =
//                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
//        polakRibierePlusSolver.setMethod(NonlinearConjugateGradientSolver.Method.POLAK_RIBIERE_PLUS);
//        polakRibierePlusSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.NO_RESTART);
//        double[] actualResult = polakRibierePlusSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

//        System.out.println("Rosenbrock Function Polak-Ribiere+ N-Step-Restart Solver:\n");
//        polakRibierePlusSolver =
//                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
//        polakRibierePlusSolver.setMethod(NonlinearConjugateGradientSolver.Method.POLAK_RIBIERE_PLUS);
//        polakRibierePlusSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.N_STEP);
//        actualResult = polakRibierePlusSolver.solve().getArray();
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Polak-Ribiere+ Gradients-Orthogonality-Check-Restart Solver:\n");
        NonlinearConjugateGradientSolver polakRibierePlusSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        polakRibierePlusSolver.setMethod(NonlinearConjugateGradientSolver.Method.POLAK_RIBIERE_PLUS);
        polakRibierePlusSolver.setRestartMethod(
                NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK
        );
        double[] actualResult = polakRibierePlusSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testHestenesStiefelSolver() {
//        System.out.println("Rosenbrock Function Hestenes-Stiefel Solver:\n");
//        NonlinearConjugateGradientSolver hestenesStiefelSolver =
//                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
//        hestenesStiefelSolver.setMethod(NonlinearConjugateGradientSolver.Method.HESTENES_STIEFEL);
//        hestenesStiefelSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.NO_RESTART);
//        double[] actualResult = hestenesStiefelSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
//        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Hestenes-Stiefel N-Step-Restart Solver:\n");
        NonlinearConjugateGradientSolver hestenesStiefelSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        hestenesStiefelSolver.setMethod(NonlinearConjugateGradientSolver.Method.HESTENES_STIEFEL);
        hestenesStiefelSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.N_STEP);
        double[] actualResult = hestenesStiefelSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Hestenes-Stiefel Gradients-Orthogonality-Check-Restart Solver:\n");
        hestenesStiefelSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        hestenesStiefelSolver.setMethod(NonlinearConjugateGradientSolver.Method.HESTENES_STIEFEL);
        hestenesStiefelSolver.setRestartMethod(
                NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK
        );
        actualResult = hestenesStiefelSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testFletcherRievesPolakRibiereSolver() {
        System.out.println("Rosenbrock Function Fletcher-Rieves-Polak-Ribiere Solver:\n");
        NonlinearConjugateGradientSolver fletcherRievesPolakRibiereSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherRievesPolakRibiereSolver.setMethod(
                NonlinearConjugateGradientSolver.Method.FLETCHER_RIEVES_POLAK_RIBIERE
        );
        fletcherRievesPolakRibiereSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.NO_RESTART);
        double[] actualResult = fletcherRievesPolakRibiereSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Fletcher-Rieves-Polak-Ribiere N-Step-Restart Solver:\n");
        fletcherRievesPolakRibiereSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherRievesPolakRibiereSolver.setMethod(
                NonlinearConjugateGradientSolver.Method.FLETCHER_RIEVES_POLAK_RIBIERE
        );
        fletcherRievesPolakRibiereSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.N_STEP);
        actualResult = fletcherRievesPolakRibiereSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Fletcher-Rieves-Polak-Ribiere Gradients-Orthogonality-Check-Restart Solver:\n");
        fletcherRievesPolakRibiereSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        fletcherRievesPolakRibiereSolver.setMethod(NonlinearConjugateGradientSolver.Method.FLETCHER_RIEVES_POLAK_RIBIERE);
        fletcherRievesPolakRibiereSolver.setRestartMethod(
                NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK
        );
        actualResult = fletcherRievesPolakRibiereSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testDaiYuanSolver() {
        System.out.println("Rosenbrock Function Dai-Yuan Solver:\n");
        NonlinearConjugateGradientSolver daiYuanSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        daiYuanSolver.setMethod(NonlinearConjugateGradientSolver.Method.DAI_YUAN);
        daiYuanSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.NO_RESTART);
        double[] actualResult = daiYuanSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Dai-Yuan N-Step-Restart Solver:\n");
        daiYuanSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        daiYuanSolver.setMethod(NonlinearConjugateGradientSolver.Method.DAI_YUAN);
        daiYuanSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.N_STEP);
        actualResult = daiYuanSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Dai-Yuan Gradients-Orthogonality-Check-Restart Solver:\n");
        daiYuanSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        daiYuanSolver.setMethod(NonlinearConjugateGradientSolver.Method.DAI_YUAN);
        daiYuanSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK);
        actualResult = daiYuanSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }

    @Test
    public void testHagerZhangSolver() {
        System.out.println("Rosenbrock Function Hager-Zhang Solver:\n");
        NonlinearConjugateGradientSolver hagerZhangSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        hagerZhangSolver.setMethod(NonlinearConjugateGradientSolver.Method.HAGER_ZHANG);
        hagerZhangSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.NO_RESTART);
        double[] actualResult = hagerZhangSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Hager-Zhang N-Step-Restart Solver:\n");
        hagerZhangSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        hagerZhangSolver.setMethod(NonlinearConjugateGradientSolver.Method.HAGER_ZHANG);
        hagerZhangSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.N_STEP);
        actualResult = hagerZhangSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Rosenbrock Function Hager-Zhang Gradients-Orthogonality-Check-Restart Solver:\n");
        hagerZhangSolver =
                new NonlinearConjugateGradientSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        hagerZhangSolver.setMethod(NonlinearConjugateGradientSolver.Method.HAGER_ZHANG);
        hagerZhangSolver.setRestartMethod(NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK);
        actualResult = hagerZhangSolver.solve().getArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
