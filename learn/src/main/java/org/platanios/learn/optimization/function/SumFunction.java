package org.platanios.learn.optimization.function;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class SumFunction extends AbstractFunction {
    private final int numberOfVariables;
    private final List<int[]> termsVariables;
    private final List<AbstractFunction> terms;

    public static class Builder {
        private final int numberOfVariables;
        private final List<int[]> termsVariables = new ArrayList<>();
        private final List<AbstractFunction> terms = new ArrayList<>();

        public Builder(int numberOfVariables) {
            this.numberOfVariables = numberOfVariables;
        }

        public Builder addTerm(AbstractFunction term, int... termVariables) {
            termsVariables.add(termVariables);
            terms.add(term);
            return this;
        }

        public SumFunction build() {
            return new SumFunction(this);
        }
    }

    private SumFunction(Builder builder) {
        numberOfVariables = builder.numberOfVariables;
        termsVariables = builder.termsVariables;
        terms = builder.terms;
    }

    public final double getValue(Vector point, int termIndex) {
        return terms.get(termIndex).computeValue(point);
    }

    @Override
    protected double computeValue(Vector point) {
        double value = 0;
        for (int term = 0; term < terms.size(); term++) {
            Vector termPoint = Vectors.build(termsVariables.get(term).length, point.type());
            termPoint.set(0, termsVariables.get(term).length - 1, point.get(termsVariables.get(term)));
            value += terms.get(term).computeValue(termPoint);
        }
        return value;
    }

    public final Vector getGradient(Vector point, int termIndex) throws NonSmoothFunctionException {
        return terms.get(termIndex).computeGradient(point);
    }

    @Override
    protected Vector computeGradient(Vector point) throws NonSmoothFunctionException {
        Vector gradient = Vectors.build(point.size(), point.type());
        for (int term = 0; term < terms.size(); term++) {
            Vector termPoint = Vectors.build(termsVariables.get(term).length, point.type());
            termPoint.set(0, termsVariables.get(term).length - 1, point.get(termsVariables.get(term)));
            Vector termGradient = Vectors.build(point.size(), point.type());
            termGradient.set(termsVariables.get(term), terms.get(term).computeGradient(termPoint));
            gradient.add(termGradient);
        }
        return gradient;
    }

    public final Matrix getHessian(Vector point, int termIndex) throws NonSmoothFunctionException {
        return terms.get(termIndex).computeHessian(point);
    }

    @Override
    protected Matrix computeHessian(Vector point) throws NonSmoothFunctionException {
        Matrix hessian = new Matrix(point.size(), point.size());
        for (int term = 0; term < terms.size(); term++) {
            Vector termPoint = Vectors.build(termsVariables.get(term).length, point.type());
            termPoint.set(0, termsVariables.get(term).length - 1, point.get(termsVariables.get(term)));
            Matrix termHessian = new Matrix(point.size(), point.size());
            termHessian.setSubMatrix(termsVariables.get(term),
                                     termsVariables.get(term),
                                     terms.get(term).computeHessian(termPoint));
            hessian.add(termHessian);
        }
        return hessian;
    }

    public int getNumberOfVariables() {
        return numberOfVariables;
    }

    public int getNumberOfTerms() {
        return terms.size();
    }

    public List<int[]> getTermsVariables() {
        return termsVariables;
    }

    public int[] getTermVariables(int termIndex) {
        return termsVariables.get(termIndex);
    }

    public List<AbstractFunction> getTerms() {
        return terms;
    }

    public AbstractFunction getTerm(int termIndex) {
        return terms.get(termIndex);
    }
}
