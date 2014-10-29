package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Classifiers {
    @SuppressWarnings("unchecked")
    public static <T extends Vector, S> Classifier<T, S> build(ObjectInputStream inputStream, ClassifierType type) throws IOException {
        return type.build(inputStream);
    }
}
