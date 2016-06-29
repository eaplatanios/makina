package makina.optimization;

import makina.math.matrix.Matrix;
import makina.math.matrix.SingularMatrixException;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.optimization.constraint.AbstractInequalityConstraint;
import makina.optimization.constraint.LinearEqualityConstraint;
import makina.optimization.function.AbstractFunction;
import makina.optimization.function.LogBarrierFunction;
import makina.optimization.function.NonSmoothFunctionException;
import makina.utilities.MathUtilities;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class NonlinearInteriorPointSolver extends AbstractIterativeSolver {
    private final List<AbstractInequalityConstraint> inequalityConstraints;
    private final InternalSolver internalSolver;
    private final AbstractFunction barrierFunction;
    private final double barrierParameter;
    private final double barrierRatioTolerance;
    private final boolean checkForBarrierParameterConvergence;

    private LinearEqualityConstraint linearEqualityConstraint = null;
    private double tau = 1.0;
    private boolean barrierConverged = false;

    private double barrierRatio;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractIterativeSolver.AbstractBuilder<T> {
        private final List<LinearEqualityConstraint> linearEqualityConstraints = new ArrayList<>();
        private final List<AbstractInequalityConstraint> inequalityConstraints = new ArrayList<>();

        private InternalSolver internalSolver = InternalSolver.NEWTON;
        private double barrierParameter = 2.0;
        private double barrierParameterTolerance = 1e-6;
        private boolean checkForBarrierParameterConvergence = true;

        protected AbstractBuilder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
//            checkForPointConvergence = false;
//            checkForObjectiveConvergence = false;
//            checkForGradientConvergence = false;
        }

        public T addLinearEqualityConstraint(LinearEqualityConstraint linearEqualityConstraint) {
            if (linearEqualityConstraint != null)
                linearEqualityConstraints.add(linearEqualityConstraint);
            return self();
        }

        public T addLinearEqualityConstraints(List<LinearEqualityConstraint> linearEqualityConstraints) {
            this.linearEqualityConstraints.addAll(linearEqualityConstraints);
            return self();
        }

        public T addInequalityConstraint(AbstractInequalityConstraint inequalityConstraint) {
            inequalityConstraints.add(inequalityConstraint);
            return self();
        }

        public T addInequalityConstraints(List<AbstractInequalityConstraint> inequalityConstraints) {
            this.inequalityConstraints.addAll(inequalityConstraints);
            return self();
        }

        public T internalSolver(InternalSolver internalSolver) {
            this.internalSolver = internalSolver;
            return self();
        }

        public T barrierParameter(double barrierParameter) {
            this.barrierParameter = barrierParameter;
            return self();
        }

        public T barrierParameterTolerance(double barrierParameterTolerance) {
            this.barrierParameterTolerance = barrierParameterTolerance;
            return self();
        }

        public T checkForBarrierParameterConvergence(boolean checkForBarrierParameterConvergence) {
            this.checkForBarrierParameterConvergence = checkForBarrierParameterConvergence;
            return self();
        }

        public NonlinearInteriorPointSolver build() {
            return new NonlinearInteriorPointSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private NonlinearInteriorPointSolver(AbstractBuilder<?> builder) {
        super(builder);
        inequalityConstraints = builder.inequalityConstraints;
        if (builder.linearEqualityConstraints.size() > 0) {
            linearEqualityConstraint = builder.linearEqualityConstraints.get(0);
            for (int constraintIndex = 1; constraintIndex < builder.linearEqualityConstraints.size(); constraintIndex++)
                linearEqualityConstraint =
                        linearEqualityConstraint.append(builder.linearEqualityConstraints.get(constraintIndex));
        }
        internalSolver = builder.internalSolver;
        barrierFunction = new LogBarrierFunction(inequalityConstraints);
        barrierParameter = builder.barrierParameter;
        barrierRatioTolerance = builder.barrierParameterTolerance;
        checkForBarrierParameterConvergence = builder.checkForBarrierParameterConvergence;
        barrierRatio = barrierParameter / tau;
    }

    @Override
    public Vector solve() {
        if (linearEqualityConstraint != null)
            try {
                currentPoint = linearEqualityConstraint.project(currentPoint);
            } catch (SingularMatrixException e) {
                logger.error("The linear equality constraint matrix is singular.", e);
            }
        double maxInequalityConstraintValue =
                inequalityConstraints.stream()
                        .map(constraint -> constraint.getValue(currentPoint))
                        .reduce(-1.0, Double::max);
        if (maxInequalityConstraintValue >= 0.0)
            findFeasiblePoint(maxInequalityConstraintValue);
        return super.solve();
    }

    @Override
    public boolean checkTerminationConditions() {
        if (super.checkTerminationConditions())
            return true;
        if (checkForBarrierParameterConvergence) {
            barrierConverged = barrierRatio <= barrierRatioTolerance;
            return barrierConverged;
        }
        return false;
    }

    @Override
    public void printIteration() {
        if (logObjectiveValue && logGradientNorm)
            logger.info("Iteration #: %10d | Func. Eval. #: %10d | Objective Value: %20s " +
                                "| Objective Change: %20s | Point Change: %20s | Gradient Norm: %20s " +
                                "| Barrier Ratio: %20s",
                        currentIteration,
                        objective.getNumberOfFunctionEvaluations(),
                        DECIMAL_FORMAT.format(currentObjectiveValue),
                        DECIMAL_FORMAT.format(objectiveChange),
                        DECIMAL_FORMAT.format(pointChange),
                        DECIMAL_FORMAT.format(gradientNorm),
                        DECIMAL_FORMAT.format(barrierRatio));
        else if (logObjectiveValue)
            logger.info("Iteration #: %10d | Func. Eval. #: %10d | Objective Value: %20s " +
                                "| Objective Change: %20s | Point Change: %20s | Barrier Ratio: %20s",
                        currentIteration,
                        objective.getNumberOfFunctionEvaluations(),
                        DECIMAL_FORMAT.format(currentObjectiveValue),
                        DECIMAL_FORMAT.format(objectiveChange),
                        DECIMAL_FORMAT.format(pointChange),
                        DECIMAL_FORMAT.format(barrierRatio));
        else if (logGradientNorm)
            logger.info("Iteration #: %10d | Func. Eval. #: %10d | Point Change: %20s | Gradient Norm: %20s " +
                                "| Barrier Ratio: %20s",
                        currentIteration,
                        objective.getNumberOfFunctionEvaluations(),
                        DECIMAL_FORMAT.format(pointChange),
                        DECIMAL_FORMAT.format(gradientNorm),
                        DECIMAL_FORMAT.format(barrierRatio));
        else
            logger.info("Iteration #: %10d | Func. Eval. #: %10d | Point Change: %20s| Barrier Ratio: %20s",
                        currentIteration,
                        objective.getNumberOfFunctionEvaluations(),
                        DECIMAL_FORMAT.format(pointChange),
                        DECIMAL_FORMAT.format(barrierRatio));
    }

    @Override
    public void printTerminationMessage() {
        super.printTerminationMessage();
        if (barrierConverged)
            logger.info("The barrier ratio, %s, was below the convergence threshold of %s.",
                        DECIMAL_FORMAT.format(barrierRatio),
                        DECIMAL_FORMAT.format(barrierRatioTolerance));
    }

    @Override
    public void performIterationUpdates() {
        previousPoint = currentPoint;
        currentPoint = internalSolver
                .solver(new BarrierObjectiveFunction(), currentPoint, linearEqualityConstraint)
                .solve();
        barrierRatio = barrierParameter / tau;
        tau = barrierParameter * tau;
        if (checkForObjectiveConvergence || logObjectiveValue) {
            previousObjectiveValue = currentObjectiveValue;
            currentObjectiveValue = new BarrierObjectiveFunction().getValue(currentPoint);
        }
        if (checkForGradientConvergence || logGradientNorm) {
            previousGradient = currentGradient;
            try {
                currentGradient = new BarrierObjectiveFunction().getGradient(currentPoint);
            } catch (NonSmoothFunctionException e) {
                e.printStackTrace();
            }
        }
    }

    private class BarrierObjectiveFunction extends AbstractFunction {
        @Override
        protected double computeValue(Vector point) {
            return tau * objective.getValue(point) + barrierFunction.getValue(point);
        }

        @Override
        protected Vector computeGradient(Vector point) throws NonSmoothFunctionException {
            return objective.getGradient(point).multInPlace(tau).addInPlace(barrierFunction.getGradient(point));
        }

        @Override
        protected Matrix computeHessian(Vector point) throws NonSmoothFunctionException {
            return objective.getHessian(point).multiplyEquals(tau).addEquals(barrierFunction.getHessian(point));
        }
    }

    private void findFeasiblePoint(double maxInequalityConstraintValue) {
        Vector initialPoint = Vectors.build(currentPoint.size() + 1, currentPoint.type());
        initialPoint.set(0, maxInequalityConstraintValue);
        initialPoint.set(1, initialPoint.size() - 1, currentPoint);
        List<AbstractInequalityConstraint> appendedConstraints =
                inequalityConstraints.stream()
                        .map(constraint -> new AppendedInequalityConstraint(
                                constraint,
                                -maxInequalityConstraintValue - MathUtilities.computeMachineEpsilonDouble()
                        ))
                        .collect(Collectors.toList());
        NonlinearInteriorPointSolver feasiblePointSolver =
                new NonlinearInteriorPointSolver.Builder(new FeasiblePointObjectiveFunction(), initialPoint)
                        .internalSolver(InternalSolver.GRADIENT_DESCENT)
                        .addLinearEqualityConstraint(linearEqualityConstraint)
                        .addInequalityConstraints(appendedConstraints)
                        .additionalCustomConvergenceCriterion(point -> point.get(0) < 0)
                        .loggingLevel(5)
                        .build();
        currentPoint = feasiblePointSolver.solve().get(1, initialPoint.size() - 1);
    }

    private class FeasiblePointObjectiveFunction extends AbstractFunction {
        @Override
        protected double computeValue(Vector point) {
            return point.get(0);
        }

        @Override
        protected Vector computeGradient(Vector point) {
            Vector gradient = Vectors.build(point.size(), point.type());
            gradient.set(0, 1.0);
            return gradient;
        }

        @Override
        protected Matrix computeHessian(Vector point) {
            return Matrix.zeros(point.size(), point.size());
        }
    }

    private class AppendedInequalityConstraint extends AbstractInequalityConstraint {
        private final AbstractInequalityConstraint inequalityConstraint;
        private final double constant;

        private AppendedInequalityConstraint(AbstractInequalityConstraint inequalityConstraint,
                                            double constant) {
            this.inequalityConstraint = inequalityConstraint;
            this.constant = constant;
        }

        @Override
        protected double computeValue(Vector point) {
            return inequalityConstraint.getValue(point.get(1, point.size() - 1)) + constant;
        }

        @Override
        protected Vector computeGradient(Vector point) throws NonSmoothFunctionException {
            Vector gradient = Vectors.build(point.size(), point.type());
            gradient.set(0, 1.0);
            gradient.set(1, gradient.size() - 1, inequalityConstraint.getGradient(point.get(1, point.size() - 1)));
            return gradient;
        }

        @Override
        protected Matrix computeHessian(Vector point) throws NonSmoothFunctionException {
            Matrix hessian = Matrix.zeros(point.size(), point.size());
            hessian.setSubMatrix(1, point.size() - 1,
                                 1, point.size() - 1,
                                 inequalityConstraint.getHessian(point.get(1, point.size() - 1)));
            return hessian;
        }
    }

    public enum InternalSolver {
        GRADIENT_DESCENT {
            @Override
            protected Solver solver(AbstractFunction objectiveFunction,
                                                              Vector currentPoint,
                                                              LinearEqualityConstraint linearEqualityConstraint) {
                return new GradientDescentSolver.Builder(objectiveFunction, currentPoint)
                        .addLinearEqualityConstraint(linearEqualityConstraint)
                        .loggingLevel(0)
                        .build();
            }
        },
        NEWTON {
            @Override
            protected Solver solver(AbstractFunction objectiveFunction,
                                                              Vector currentPoint,
                                                              LinearEqualityConstraint linearEqualityConstraint) {
                return new NewtonSolver.Builder(objectiveFunction, currentPoint)
                        .addLinearEqualityConstraint(linearEqualityConstraint)
                        .loggingLevel(0)
                        .build();
            }
        };

        protected abstract Solver solver(AbstractFunction objectiveFunction,
                                         Vector currentPoint,
                                         LinearEqualityConstraint linearEqualityConstraint);
    }
}
