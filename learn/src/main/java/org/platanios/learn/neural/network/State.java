package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.math.matrix.Vectors;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * // TODO: Add support for variables initialization method.
 *
 * @author Emmanouil Antonios Platanios
 */
public class State {
    private final Map<Variable, Vector> variableValues = new HashMap<>();

    private final VectorType vectorType;

    public State() {
        this(new HashSet<>(), VectorType.DENSE);
    }

    public State(Set<Variable> variables) {
        this(variables, VectorType.DENSE);
    }

    public State(Set<Variable> variables, VectorType vectorType) {
        this.vectorType = vectorType;
        for (Variable variable : variables)
            variableValues.put(variable, Vectors.build(variable.size, vectorType));
    }

    public VectorType vectorType() {
        return vectorType;
    }

    public void set(Variable variable, Vector value) {
        variableValues.put(variable, value);
    }

    public void set(int id, Vector value) {
        variableValues.put(Variables.get(id), value);
    }

    public void set(String name, Vector value) {
        variableValues.put(Variables.get(name), value);
    }

    public Vector get(Variable variable) {
        return variableValues.getOrDefault(variable, null);
    }

    public Vector get(int id) {
        return variableValues.getOrDefault(Variables.get(id), null);
    }

    public Vector get(String name) {
        return variableValues.getOrDefault(Variables.get(name), null);
    }
}
