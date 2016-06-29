package makina.learn.neural.network;

import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Network {
    private final VariablesManager variablesManager;
    private final NetworkState state;
    private final Layer[] layers;
    private final Variable[] variables;
    private final Variable[] parameters;
    private final Layer[] inputLayers;
    private final Layer outputLayer;
    private final int inputSize;
    private final int outputSize;

    Network(NetworkBuilder builder) {
        variablesManager = builder.variablesManager;
        state = new NetworkState(variablesManager);
        layers = builder.layers.stream().toArray(Layer[]::new);
        variables = builder.variables.stream().toArray(Variable[]::new);
        parameters = builder.parameters.stream().toArray(Variable[]::new);
        inputLayers = builder.inputLayers.stream().toArray(Layer[]::new);
        outputLayer = builder.outputLayer;
        inputSize = builder.inputLayers.size();
        outputSize = builder.outputLayer.outputSize();
    }

    public String[] parameters() {
        String[] parameters = new String[this.parameters.length];
        for (int parameterIndex = 0; parameterIndex < parameters.length; parameterIndex++)
            parameters[parameterIndex] = this.parameters[parameterIndex].name();
        return parameters;
    }

    public String[] variables() {
        String[] variables = new String[this.parameters.length];
        for (int variableIndex = 0; variableIndex < variables.length; variableIndex++)
            variables[variableIndex] = this.variables[variableIndex].name();
        return variables;
    }

    public void set(int variableId, Vector value) {
        state.set(variableId, value);
    }

    public void set(String variableName, Vector value) {
        state.set(variableName, value);
    }

    private void set(Variable variable, Vector value) {
        state.set(variable, value);
    }

    public Vector get(int variableId) {
        return state.get(variableId);
    }

    public Vector get(String variableName) {
        return state.get(variableName);
    }

    private void get(Variable variable) {
        state.get(variable);
    }

    public Vector value() {
        return outputLayer.value(state);
    }

    public Matrix recursiveGradient(int variableId) {
        return recursiveGradient(variablesManager.get(variableId));
    }

    public Matrix recursiveGradient(String variableName) {
        return recursiveGradient(variablesManager.get(variableName));
    }

    private Matrix recursiveGradient(Variable variable) {
        return outputLayer.recursiveGradient(state, variable);
    }

    public Matrix[] recursiveGradient(int... variableIds) {
        Variable[] variables = new Variable[variableIds.length];
        for (int variableIndex = 0; variableIndex < variableIds.length; variableIndex++)
            variables[variableIndex] = variablesManager.get(variableIds[variableIndex]);
        return recursiveGradient(variables);
    }

    public Matrix[] recursiveGradient(String... variableNames) {
        Variable[] variables = new Variable[variableNames.length];
        for (int variableIndex = 0; variableIndex < variableNames.length; variableIndex++)
            variables[variableIndex] = variablesManager.get(variableNames[variableIndex]);
        return recursiveGradient(variables);
    }

    private Matrix[] recursiveGradient(Variable[] variables) {
        return outputLayer.recursiveGradient(state, variables);
    }

    public Matrix gradient(int variableId) {
        return gradient(variablesManager.get(variableId));
    }

    public Matrix gradient(String variableName) {
        return gradient(variablesManager.get(variableName));
    }

    private Matrix gradient(Variable variable) {
        Matrix gradient = Matrix.zeros(outputSize, variable.size());
        for (Layer layer : layers)
            layer.resetForwardGradient();
        outputLayer.backPropagateGradient(state, outputLayer.localGradient(state, outputLayer.outputVariable()));
        int currentBackPropagationIndex = layers.length - 1;
        while (currentBackPropagationIndex >= 0)
            for (int layerIndex = currentBackPropagationIndex; layerIndex > -1; layerIndex--) {
                Matrix layerGradient = layers[layerIndex].gradient(state, variable);
                if (layerGradient != null) {
                    gradient.addEquals(layerGradient);
                    Layer swapLayer = layers[layerIndex];
                    layers[layerIndex] = layers[currentBackPropagationIndex];
                    layers[currentBackPropagationIndex] = swapLayer;
                    layerIndex = currentBackPropagationIndex--;
                }
            }
        return gradient;
    }

    public Matrix[] gradient(int... variableIds) {
        Variable[] variables = new Variable[variableIds.length];
        for (int variableIndex = 0; variableIndex < variableIds.length; variableIndex++)
            variables[variableIndex] = variablesManager.get(variableIds[variableIndex]);
        return gradient(variables);
    }

    public Matrix[] gradient(String... variableNames) {
        Variable[] variables = new Variable[variableNames.length];
        for (int variableIndex = 0; variableIndex < variableNames.length; variableIndex++)
            variables[variableIndex] = variablesManager.get(variableNames[variableIndex]);
        return gradient(variables);
    }

    private Matrix[] gradient(Variable... variables) {
        Matrix[] gradients = new Matrix[variables.length];
        for (int variableIndex = 0; variableIndex < variables.length; variableIndex++)
            gradients[variableIndex] = Matrix.zeros(outputSize, variables[variableIndex].size());
        for (Layer layer : layers)
            layer.resetForwardGradient();
        outputLayer.backPropagateGradient(state, outputLayer.localGradient(state, outputLayer.outputVariable()));
        int currentBackPropagationIndex = layers.length - 1;
        while (currentBackPropagationIndex >= 0)
            for (int layerIndex = currentBackPropagationIndex; layerIndex > -1; layerIndex--) {
                Matrix[] layerGradients = layers[layerIndex].gradient(state, variables);
                if (layerGradients != null) {
                    for (int gradientIndex = 0; gradientIndex < gradients.length; gradientIndex++)
                        gradients[gradientIndex].addEquals(layerGradients[gradientIndex]);
                    Layer swapLayer = layers[layerIndex];
                    layers[layerIndex] = layers[currentBackPropagationIndex];
                    layers[currentBackPropagationIndex] = swapLayer;
                    layerIndex = currentBackPropagationIndex--;
                }
            }
        return gradients;
    }

    public ObjectiveFunction getObjectiveFunction(LossFunction lossFunction, int... variableIds) {
        Variable[] variables = new Variable[variableIds.length];
        for (int variableIndex = 0; variableIndex < variableIds.length; variableIndex++)
            variables[variableIndex] = variablesManager.get(variableIds[variableIndex]);
        return new ObjectiveFunction(lossFunction, variables);
    }

    public ObjectiveFunction getObjectiveFunction(LossFunction lossFunction, String... variableNames) {
        Variable[] variables = new Variable[variableNames.length];
        for (int variableIndex = 0; variableIndex < variableNames.length; variableIndex++)
            variables[variableIndex] = variablesManager.get(variableNames[variableIndex]);
        return new ObjectiveFunction(lossFunction, variables);
    }

    public class ObjectiveFunction extends AbstractFunction {
        private final LossFunction lossFunction;
        private final Variable[] variables;

        private ObjectiveFunction(LossFunction lossFunction, Variable[] variables) {
            if (lossFunction.inputSize() != outputSize)
                throw new IllegalArgumentException("The input size of the loss function must match " +
                                                           "the output size of the network.");
            this.lossFunction = lossFunction;
            this.variables = variables;
        }

        @Override
        protected double computeValue(Vector point) {
            int vectorIndex = 0;
            for (Variable variable : variables) {
                set(variable, point.get(vectorIndex, vectorIndex + variable.size() - 1));
                vectorIndex += variable.size();
            }
            return lossFunction.value(value());
        }

        @Override
        protected Vector computeGradient(Vector point) {
            int vectorIndex = 0;
            for (Variable variable : variables) {
                set(variable, point.get(vectorIndex, vectorIndex + variable.size() - 1));
                vectorIndex += variable.size();
            }
            Vector lossFunctionGradient = lossFunction.gradient(value());
            Matrix[] networkGradients = gradient(variables);
            Vector gradient = Vectors.build(point.size(), point.type());
            vectorIndex = 0;
            for (Matrix networkGradient : networkGradients)
                gradient.set(vectorIndex,
                             vectorIndex + networkGradient.getColumnDimension() - 1,
                             lossFunctionGradient.transMult(networkGradient));
            return gradient;
        }
    }
}
