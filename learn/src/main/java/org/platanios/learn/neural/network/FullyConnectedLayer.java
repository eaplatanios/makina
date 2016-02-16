package org.platanios.learn.neural.network;

import com.google.common.collect.Sets;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FullyConnectedLayer extends SingleInputLayer {
    private final int numberOfHiddenUnits;
    private final boolean includeBiasTerm;
    private final MatrixVariable weights;
    private final VectorVariable bias;

    FullyConnectedLayer(Layer inputLayer, int numberOfHiddenUnits) {
        this(inputLayer, numberOfHiddenUnits, true);
    }

    FullyConnectedLayer(Layer inputLayer, int numberOfHiddenUnits, boolean includeBiasTerm) {
        super(inputLayer, numberOfHiddenUnits);
        this.numberOfHiddenUnits = numberOfHiddenUnits;
        this.includeBiasTerm = includeBiasTerm;
        this.weights = Variables.matrixVariable(numberOfHiddenUnits, inputSize);
        if (includeBiasTerm)
            this.bias = Variables.vectorVariable(numberOfHiddenUnits);
        else
            this.bias = null;
    }

    FullyConnectedLayer(Layer inputLayer, int numberOfHiddenUnits, String weightsVariableName) {
        this(inputLayer, numberOfHiddenUnits, weightsVariableName, null);
    }

    FullyConnectedLayer(Layer inputLayer,
                        int numberOfHiddenUnits,
                        String weightsVariableName,
                        String biasVariableName) {
        super(inputLayer, numberOfHiddenUnits);
        this.numberOfHiddenUnits = numberOfHiddenUnits;
        this.includeBiasTerm = biasVariableName != null;
        this.weights = Variables.matrixVariable(weightsVariableName, numberOfHiddenUnits, inputSize);
        if (includeBiasTerm)
            this.bias = Variables.vectorVariable(biasVariableName, numberOfHiddenUnits);
        else
            this.bias = null;
    }

    @Override
    public Set<Variable> parameters() {
        if (includeBiasTerm)
            return Sets.newHashSet(weights, bias);
        else
            return Sets.newHashSet(weights);
    }

    @Override
    public Vector computeValue(State state) {
        Vector inputValue = inputLayer.value(state);
        Vector outputValue = Vectors.build(numberOfHiddenUnits, inputValue.type());
        Vector weights = state.get(this.weights);
        Vector bias = null;
        if (includeBiasTerm)
            bias = state.get(this.bias);
        for (int i = 0; i < numberOfHiddenUnits; i++) {
            double sum = 0.0;
            if (includeBiasTerm)
                sum += bias.get(i);
            for (int j = 0; j < inputSize; j++)
                sum += weights.get(i + j * numberOfHiddenUnits) * inputValue.get(j);
            outputValue.set(i, sum);
        }
        return outputValue;
    }

    @Override
    protected Matrix selfGradient(State state, Variable variable) {
        if (variable.equals(weights)) {
            Vector inputValue = inputLayer.value(state);
            Matrix gradient = Matrix.zeros(numberOfHiddenUnits, weights.size);
            for (int i = 0; i < numberOfHiddenUnits; i++)
                for (int j = 0; j < inputSize; j++)
                    gradient.setElement(i, i + j * numberOfHiddenUnits, inputValue.get(j));
            return gradient;
        } else if (includeBiasTerm && variable.equals(bias)) {
            return Matrix.identity(numberOfHiddenUnits);
        } else {
            return Matrix.zeros(numberOfHiddenUnits, variable.size);
        }
    }
}
