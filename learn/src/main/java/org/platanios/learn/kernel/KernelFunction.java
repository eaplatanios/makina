package org.platanios.learn.kernel;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface KernelFunction<T> {
    double getValue(T instance1, T instance2);
}
