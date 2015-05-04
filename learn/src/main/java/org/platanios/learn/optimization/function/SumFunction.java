package org.platanios.learn.optimization.function;

import org.apache.commons.math3.analysis.function.Abs;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidObjectException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SumFunction extends AbstractFunction {
    protected final int numberOfVariables;
    protected final List<int[]> termsVariables;
    protected final List<AbstractFunction> terms;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final int numberOfVariables;
        protected final List<int[]> termsVariables;
        protected final List<AbstractFunction> terms;

        protected AbstractBuilder(int numberOfVariables) {
            this.numberOfVariables = numberOfVariables;
            this.termsVariables = new ArrayList<>();
            this.terms = new ArrayList<>();
        }

        protected AbstractBuilder(
                int numberOfVariables,
                List<AbstractFunction> terms,
                List<int[]> termsVariables) {
            this.numberOfVariables = numberOfVariables;
            this.termsVariables = termsVariables;
            this.terms = terms;
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

        private Builder(
                int numberOfVariables,
                List<AbstractFunction> terms,
                List<int[]> termVariables)
        {
            super(numberOfVariables, terms, termVariables);
        }

        @Override
        protected Builder self() {
            return this;
        }

        public static SumFunction build(InputStream inputStream, boolean includeType) throws IOException {
            if (includeType) {
                FunctionType functionType = FunctionType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
                if (functionType != FunctionType.SumFunction) {
                    throw new InvalidObjectException("The stored function is of type " + functionType.name() + "!");
                }
            }

            int numberOfVariables = UnsafeSerializationUtilities.readInt(inputStream);

            int termsSize = UnsafeSerializationUtilities.readInt(inputStream);
            ArrayList<AbstractFunction> terms = new ArrayList<>(termsSize);
            for (int i = 0; i < termsSize; ++i) {
                terms.add(AbstractFunction.build(inputStream));
            }

            int termsVariablesSize = UnsafeSerializationUtilities.readInt(inputStream);
            ArrayList<int[]> termsVariables = new ArrayList<>();
            for (int i = 0; i < termsVariablesSize; ++i) {
                int termVariablesSize = UnsafeSerializationUtilities.readInt(inputStream);
                int[] termVariables = new int[termVariablesSize];
                for (int j = 0; j < termVariables.length; ++j) {
                    termVariables[j] = UnsafeSerializationUtilities.readInt(inputStream);
                }
                termsVariables.add(termVariables);
            }

            Builder builder = new Builder(numberOfVariables, terms, termsVariables);
            return builder.build();
        }
    }

    protected SumFunction(AbstractBuilder<?> builder) {
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

    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException {
        if (includeType) {
            UnsafeSerializationUtilities.writeInt(outputStream, FunctionType.SumFunction.ordinal());
        }
        UnsafeSerializationUtilities.writeInt(outputStream, this.numberOfVariables);
        UnsafeSerializationUtilities.writeInt(outputStream, this.terms.size());
        for (AbstractFunction term : this.terms) {
            term.write(outputStream, true);
        }
        UnsafeSerializationUtilities.writeInt(outputStream, this.termsVariables.size());
        for (int[] termVariables : this.termsVariables) {
            UnsafeSerializationUtilities.writeInt(outputStream, termVariables.length);
            for (int variable : termVariables) {
                UnsafeSerializationUtilities.writeInt(outputStream, variable);
            }
        }
    }

}
