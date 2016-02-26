package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.math.matrix.Vectors;

import java.util.HashMap;
import java.util.Map;

/**
 * // TODO: Add support for variables initialization method.
 *
 * @author Emmanouil Antonios Platanios
 */
class NetworkState {
    private final Map<Variable, Vector> variableValues = new HashMap<>();

    private final VariablesManager variablesManager;
    private final VectorType vectorType;

    NetworkState(VariablesManager variablesManager) {
        this(variablesManager, VectorType.DENSE);
    }

    NetworkState(VariablesManager variablesManager, VectorType vectorType) {
        this.variablesManager = variablesManager;
        this.vectorType = vectorType;
        for (Variable variable : variablesManager.variables())
            variableValues.put(variable, Vectors.build(variable.size, vectorType));
    }

    VectorType vectorType() {
        return vectorType;
    }

    void set(Variable variable, Vector value) {
        variableValues.put(variable, value);
    }

    void set(int id, Vector value) {
        variableValues.put(variablesManager.get(id), value);
    }

    void set(String name, Vector value) {
        variableValues.put(variablesManager.get(name), value);
    }

    Vector get(Variable variable) {
        return variableValues.getOrDefault(variable, null);
    }

    Vector get(int id) {
        return variableValues.getOrDefault(variablesManager.get(id), null);
    }

    Vector get(String name) {
        return variableValues.getOrDefault(variablesManager.get(name), null);
    }
}
