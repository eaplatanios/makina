package org.platanios.learn.optimization.function;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SumFunction extends AbstractFunction {
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

    public final double getValue(Vector point, int term) {
        Vector termPoint = Vectors.build(termsVariables.get(term).length, point.type());
        termPoint.set(0, termsVariables.get(term).length - 1, point.get(termsVariables.get(term)));
        return terms.get(term).computeValue(termPoint);
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

    public final Vector getGradient(Vector point, int term) {
        Vector termPoint = Vectors.build(termsVariables.get(term).length, point.type());
        termPoint.set(0, termsVariables.get(term).length - 1, point.get(termsVariables.get(term)));
        return terms.get(term).computeGradient(termPoint);
    }

    @Override
    protected Vector computeGradient(Vector point) {
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

    public final Matrix getHessian(Vector point, int term) {
        Vector termPoint = Vectors.build(termsVariables.get(term).length, point.type());
        termPoint.set(0, termsVariables.get(term).length - 1, point.get(termsVariables.get(term)));
        return terms.get(term).computeHessian(termPoint);
    }

    @Override
    protected Matrix computeHessian(Vector point) {
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

    public List<int[]> getTermsVariables() {
        return termsVariables;
    }

    public List<AbstractFunction> getTerms() {
        return terms;
    }
}
