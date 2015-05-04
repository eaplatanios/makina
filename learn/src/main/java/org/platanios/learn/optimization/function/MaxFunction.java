package org.platanios.learn.optimization.function;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;
import sun.misc.Unsafe;

import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidObjectException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MaxFunction extends AbstractFunction {
    private final int numberOfVariables;
    private final List<int[]> functionTermVariables;
    private final List<AbstractFunction> functionTerms;
    private final List<Double> constantTerms;

    public static class Builder {
        private final int numberOfVariables;
        private final List<int[]> functionTermVariables;
        private final List<AbstractFunction> functionTerms;
        private final List<Double> constantTerms;

        public Builder(int numberOfVariables) {
            this.numberOfVariables = numberOfVariables;
            this.functionTermVariables = new ArrayList<>();
            this.functionTerms = new ArrayList<>();
            this.constantTerms = new ArrayList<>();
        }

        private Builder(
                int numberOfVariables,
                List<AbstractFunction> functionTerms,
                List<Double> constantTerms,
                List<int[]> functionTermVariables) {
            this.numberOfVariables = numberOfVariables;
            this.functionTermVariables = functionTermVariables;
            this.functionTerms = functionTerms;
            this.constantTerms = constantTerms;
        }

        public Builder addFunctionTerm(AbstractFunction functionTerm) {
            functionTermVariables.add(null);
            functionTerms.add(functionTerm);
            return this;
        }

        public Builder addFunctionTerm(AbstractFunction functionTerm, int... termVariables) {
            functionTermVariables.add(termVariables);
            functionTerms.add(functionTerm);
            return this;
        }

        public Builder addConstantTerm(double constantTerm) {
            constantTerms.add(constantTerm);
            return this;
        }

        public MaxFunction build() {
            return new MaxFunction(this);
        }
        public static MaxFunction build(InputStream inputStream, boolean includeType) throws IOException {
            if (includeType) {
                FunctionType functionType = FunctionType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
                if (functionType != FunctionType.MaxFunction) {
                    throw new InvalidObjectException("The stored function is of type " + functionType.name() + "!");
                }
            }

            int numberOfVariables = UnsafeSerializationUtilities.readInt(inputStream);

            int constantTermsSize = UnsafeSerializationUtilities.readInt(inputStream);
            ArrayList<Double> constantTerms = new ArrayList<>(constantTermsSize);
            for (int i = 0; i < constantTermsSize; ++i) {
                constantTerms.add(UnsafeSerializationUtilities.readDouble(inputStream));
            }

            int functionTermsSize = UnsafeSerializationUtilities.readInt(inputStream);
            ArrayList<AbstractFunction> functionTerms = new ArrayList<>(functionTermsSize);
            for (int i = 0; i < functionTermsSize; ++i) {
                functionTerms.add(AbstractFunction.build(inputStream));
            }

            int functionTermVariablesSize = UnsafeSerializationUtilities.readInt(inputStream);
            ArrayList<int[]> functionTermVariables = new ArrayList<>();
            for (int i = 0; i < functionTermVariablesSize; ++i) {
                int termVariablesSize = UnsafeSerializationUtilities.readInt(inputStream);
                int[] termVariables = new int[termVariablesSize];
                for (int j = 0; j < termVariables.length; ++j) {
                    termVariables[j] = UnsafeSerializationUtilities.readInt(inputStream);
                }
                functionTermVariables.add(termVariables);
            }

            Builder builder = new Builder(numberOfVariables, functionTerms, constantTerms, functionTermVariables);
            return builder.build();
        }
    }

    private MaxFunction(Builder builder) {
        numberOfVariables = builder.numberOfVariables;
        functionTermVariables = builder.functionTermVariables;
        functionTerms = builder.functionTerms;
        constantTerms = builder.constantTerms;
    }

    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException {
        if (includeType) {
            UnsafeSerializationUtilities.writeInt(outputStream, FunctionType.MaxFunction.ordinal());
        }
        UnsafeSerializationUtilities.writeInt(outputStream, this.numberOfVariables);
        UnsafeSerializationUtilities.writeInt(outputStream, this.constantTerms.size());
        for(double constantTerm : this.constantTerms) {
            UnsafeSerializationUtilities.writeDouble(outputStream, constantTerm);
        }
        UnsafeSerializationUtilities.writeInt(outputStream, this.functionTerms.size());
        for (AbstractFunction term : this.functionTerms) {
            term.write(outputStream, true);
        }
        UnsafeSerializationUtilities.writeInt(outputStream, this.functionTermVariables.size());
        for (int[] termVariables : this.functionTermVariables) {
            UnsafeSerializationUtilities.writeInt(outputStream, termVariables.length);
            for (int variable : termVariables) {
                UnsafeSerializationUtilities.writeInt(outputStream, variable);
            }
        }
    }

    public final double getValue(Vector point, int termIndex) {
        return functionTerms.get(termIndex).computeValue(point);
    }

    @Override
    protected double computeValue(Vector point) {
        double value = -Double.MAX_VALUE;
        for (int term = 0; term < functionTerms.size(); term++) {
            if (functionTermVariables.get(term) != null) {
                Vector termPoint = Vectors.build(functionTermVariables.get(term).length, point.type());
                termPoint.set(0, functionTermVariables.get(term).length - 1, point.get(functionTermVariables.get(term)));
                value = Math.max(value, functionTerms.get(term).computeValue(termPoint));
            } else {
                value = Math.max(value, functionTerms.get(term).computeValue(point));
            }
        }
        for (Double constantTerm : constantTerms)
            value = Math.max(value, constantTerm);
        return value;
    }

    public final Vector getGradient(Vector point, int termIndex) throws NonSmoothFunctionException {
        return functionTerms.get(termIndex).computeGradient(point);
    }

    @Override
    protected Vector computeGradient(Vector point) throws NonSmoothFunctionException {
        throw new NonSmoothFunctionException("The max function is not differentiable!");
    }

    public final Matrix getHessian(Vector point, int termIndex) throws NonSmoothFunctionException {
        return functionTerms.get(termIndex).computeHessian(point);
    }

    @Override
    protected Matrix computeHessian(Vector point) throws NonSmoothFunctionException {
        throw new NonSmoothFunctionException("The max function is not differentiable!");
    }

    public int getNumberOfVariables() {
        return numberOfVariables;
    }

    public int getNumberOfTerms() {
        return functionTerms.size();
    }

    public List<int[]> getFunctionTermVariables() {
        return functionTermVariables;
    }

    public int[] getFunctionTermVariables(int functionTermIndex) {
        return functionTermVariables.get(functionTermIndex);
    }

    public List<AbstractFunction> getFunctionTerms() {
        return functionTerms;
    }

    public AbstractFunction getFunctionTerm(int functionTermIndex) {
        return functionTerms.get(functionTermIndex);
    }

    public List<Double> getConstantTerms() {
        return constantTerms;
    }

    public double getConstantTerm(int constantTermIndex) {
        return constantTerms.get(constantTermIndex);
    }
}
