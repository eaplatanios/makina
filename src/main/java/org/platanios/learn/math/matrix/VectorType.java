package org.platanios.learn.math.matrix;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum VectorType {
    DENSE {
        @Override
        public DenseVector createVector(int size, double initialValue) {
            return new DenseVector(size, initialValue);
        }
    },
    SPARSE {
        @Override
        public DenseVector createVector(int size, double initialValue) {
            throw new NotImplementedException();
        }
    };

    public abstract Vector createVector(int size, double initialValue);
}
