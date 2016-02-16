package org.platanios.learn.neural.network;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Variables {
    private static AtomicInteger idCounter = new AtomicInteger(0);
    private static Map<Integer, Variable> variableIdsMap = new HashMap<>();
    private static Map<String, Variable> variableNamesMap = new HashMap<>();

    public static int id() {
        return idCounter.getAndIncrement();
    }

    public static VectorVariable vectorVariable(int size) {
        VectorVariable variable = new VectorVariable(id(), size);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    public static VectorVariable vectorVariable(String name, int size) {
        if (variableNamesMap.keySet().contains(name))
            throw new IllegalArgumentException("Each variable should have a unique name.");
        VectorVariable variable = new VectorVariable(id(), name, size);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    public static MatrixVariable matrixVariable(int rowDimension, int columnDimension) {
        MatrixVariable variable = new MatrixVariable(id(), rowDimension, columnDimension);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    public static MatrixVariable matrixVariable(String name, int rowDimension, int columnDimension) {
        if (variableNamesMap.keySet().contains(name))
            throw new IllegalArgumentException("Each variable should have a unique name.");
        MatrixVariable variable = new MatrixVariable(id(), name, rowDimension, columnDimension);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    public static LayerVariable layerVariable(Layer layer) {
        LayerVariable variable = new LayerVariable(id(), layer);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    public static LayerVariable layerVariable(String name, Layer layer) {
        if (variableNamesMap.keySet().contains(name))
            throw new IllegalArgumentException("Each variable should have a unique name.");
        LayerVariable variable = new LayerVariable(id(), name, layer);
        variableIdsMap.put(variable.id(), variable);
        variableNamesMap.put(variable.name(), variable);
        return variable;
    }

    public static Variable get(int id) {
        return variableIdsMap.getOrDefault(id, null);
    }

    public static Variable get(String name) {
        return variableNamesMap.getOrDefault(name, null);
    }
}
