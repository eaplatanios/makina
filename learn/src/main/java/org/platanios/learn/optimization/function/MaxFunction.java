package org.platanios.learn.optimization.function;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.analysis.function.Max;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

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
        private final List<int[]> functionTermVariables = new ArrayList<>();
        private final List<AbstractFunction> functionTerms = new ArrayList<>();
        private final List<Double> constantTerms = new ArrayList<>();

        public Builder(int numberOfVariables) {
            this.numberOfVariables = numberOfVariables;
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
    }

    private MaxFunction(Builder builder) {
        numberOfVariables = builder.numberOfVariables;
        functionTermVariables = builder.functionTermVariables;
        functionTerms = builder.functionTerms;
        constantTerms = builder.constantTerms;
    }

    @Override
    public boolean equals(Object other) {

        if (other == this) {
            return true;
        }

        if (!super.equals(other)) {
            return false;
        }

        if (!(other instanceof MaxFunction)) {
            return false;
        }

        MaxFunction rhs = (MaxFunction) other;

        return new EqualsBuilder()
                .append(this.constantTerms, rhs.constantTerms)
                .append(this.functionTerms, rhs.functionTerms)
                .append(this.functionTermVariables, rhs.functionTermVariables)
                .append(this.numberOfVariables, rhs.numberOfVariables)
                .isEquals();

    }

    @Override
    public int hashCode() {

        return new HashCodeBuilder(71, 29)
                .append(super.hashCode())
                .append(this.constantTerms)
                .append(this.functionTerms)
                .append(this.functionTermVariables)
                .append(this.numberOfVariables)
                .toHashCode();

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
