package org.platanios.optimization.function;

import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SumFunction extends AbstractFunction {
    protected final int numberOfVariables;
    protected final List<int[]> termVariables;
    protected final List<AbstractFunction> terms;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final int numberOfVariables;
        protected final List<int[]> termsVariables = new ArrayList<>();
        protected final List<AbstractFunction> terms = new ArrayList<>();

        protected AbstractBuilder(int numberOfVariables) {
            this.numberOfVariables = numberOfVariables;
        }

        public T addTerm(AbstractFunction term, int... termVariables) {
            termsVariables.add(termVariables);
            terms.add(term);
            return self();
        }

        public SumFunction build() {
            return new SumFunction(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(int numberOfVariables) {
            super(numberOfVariables);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected SumFunction(AbstractBuilder<?> builder) {
        numberOfVariables = builder.numberOfVariables;
        termVariables = builder.termsVariables;
        terms = builder.terms;
    }

    public final double getValue(Vector point, int termIndex) {
        return terms.get(termIndex).computeValue(point);
    }

    @Override
    protected double computeValue(Vector point) {
        double value = 0;
        for (int term = 0; term < terms.size(); term++) {
            Vector termPoint = Vectors.build(termVariables.get(term).length, point.type());
            termPoint.set(0, termVariables.get(term).length - 1, point.get(termVariables.get(term)));
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
            Vector termPoint = Vectors.build(termVariables.get(term).length, point.type());
            termPoint.set(0, termVariables.get(term).length - 1, point.get(termVariables.get(term)));
            Vector termGradient = Vectors.build(point.size(), point.type());
            termGradient.set(termVariables.get(term), terms.get(term).computeGradient(termPoint));
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
            Vector termPoint = Vectors.build(termVariables.get(term).length, point.type());
            termPoint.set(0, termVariables.get(term).length - 1, point.get(termVariables.get(term)));
            Matrix termHessian = new Matrix(point.size(), point.size());
            termHessian.setSubMatrix(termVariables.get(term),
                                     termVariables.get(term),
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

    public List<int[]> getTermVariables() {
        return termVariables;
    }

    public int[] getTermVariables(int termIndex) {
        return termVariables.get(termIndex);
    }

    public List<AbstractFunction> getTerms() {
        return terms;
    }

    public AbstractFunction getTerm(int termIndex) {
        return terms.get(termIndex);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        SumFunction that = (SumFunction) other;

        if (!super.equals(that))
            return false;
        if (numberOfVariables != that.numberOfVariables)
            return false;
        if (!termVariables.equals(that.termVariables))
            return false;
        if (!terms.equals(that.terms))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + numberOfVariables;
        result = 31 * result + termVariables.hashCode();
        result = 31 * result + terms.hashCode();
        return result;
    }
}
