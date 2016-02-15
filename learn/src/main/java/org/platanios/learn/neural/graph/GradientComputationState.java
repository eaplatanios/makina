package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.Matrix;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientComputationState {
    private int step = 0;
    private Map<Vertex, Matrix> computedGradients = new HashMap<>();
}
