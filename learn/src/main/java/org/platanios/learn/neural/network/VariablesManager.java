package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Vector;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Emmanouil Antonios Platanios
 */
class VariablesManager {
    private final AtomicInteger idCounter = new AtomicInteger(0);
    private final Map<Integer, Variable> variableIdsMap = new HashMap<>();
    private final Map<String, Variable> variableNamesMap = new HashMap<>();

    VariablesManager() { }

    private int id() {
        return idCounter.getAndIncrement();
    }

    Set<Variable> variables() {
        return new HashSet<>(variableIdsMap.values());
    }

    ConstantVectorVariable constantVariable(Vector value) {
        ConstantVectorVariable variable = new ConstantVectorVariable(id(), value);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    VectorVariable vectorVariable(int size) {
        VectorVariable variable = new VectorVariable(id(), size);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    VectorVariable vectorVariable(String name, int size) {
        if (variableNamesMap.keySet().contains(name))
            throw new IllegalArgumentException("Each variable should have a unique name.");
        VectorVariable variable = new VectorVariable(id(), name, size);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    MatrixVariable matrixVariable(int rowDimension, int columnDimension) {
        MatrixVariable variable = new MatrixVariable(id(), rowDimension, columnDimension);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    MatrixVariable matrixVariable(String name, int rowDimension, int columnDimension) {
        if (variableNamesMap.keySet().contains(name))
            throw new IllegalArgumentException("Each variable should have a unique name.");
        MatrixVariable variable = new MatrixVariable(id(), name, rowDimension, columnDimension);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    LayerVariable layerVariable(Layer layer) {
        LayerVariable variable = new LayerVariable(id(), layer);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    LayerVariable layerVariable(String name, Layer layer) {
        if (variableNamesMap.keySet().contains(name))
            throw new IllegalArgumentException("Each variable should have a unique name.");
        LayerVariable variable = new LayerVariable(id(), name, layer);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    Variable get(int id) {
        return variableIdsMap.getOrDefault(id, null);
    }

    Variable get(String name) {
        return variableNamesMap.getOrDefault(name, null);
    }
}
