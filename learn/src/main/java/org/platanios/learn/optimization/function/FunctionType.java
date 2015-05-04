package org.platanios.learn.optimization.function;

import java.io.IOException;
import java.io.InputStream;

/**
 * This enumeration contains the different types of functions that are supported. Each type also contains methods that can
 * be called to build functions of the corresponding type.
 *
 * @author Dan Schwartz
 */
public enum FunctionType {

    LinearFunction {
        @Override
        public org.platanios.learn.optimization.function.LinearFunction buildFunction(
                InputStream inputStream, boolean includeType) throws IOException {
            return org.platanios.learn.optimization.function.LinearFunction.read(inputStream, includeType);
        }
    },
    LinearLeastSquaresFunction {
        public org.platanios.learn.optimization.function.LinearLeastSquaresFunction buildFunction(
                InputStream inputStream, boolean includeType) throws IOException {
            return org.platanios.learn.optimization.function.LinearLeastSquaresFunction.read(inputStream, includeType);
        }
    },
    MaxFunction {
        public org.platanios.learn.optimization.function.MaxFunction buildFunction(
                InputStream inputStream, boolean includeType) throws IOException {
            return org.platanios.learn.optimization.function.MaxFunction.Builder.build(inputStream, includeType);
        }
    },
    QuadraticFunction {
        public org.platanios.learn.optimization.function.QuadraticFunction buildFunction(
                InputStream inputStream, boolean includeType) throws IOException {
            return org.platanios.learn.optimization.function.QuadraticFunction.read(inputStream, includeType);
        }
    },
    SumFunction {
        public org.platanios.learn.optimization.function.SumFunction buildFunction(
                InputStream inputStream, boolean includeType) throws IOException {
            return org.platanios.learn.optimization.function.SumFunction.Builder.build(inputStream, includeType);
        }
    };

    public abstract AbstractFunction buildFunction(InputStream inputStream, boolean includeType) throws IOException;

}
