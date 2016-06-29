package makina.learn.neural.network;

import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.utilities.ArrayUtilities;

/**
 * @author Emmanouil Antonios Platanios
 */
class ElementwiseMultiplicationLayer extends MultiInputLayer {
    ElementwiseMultiplicationLayer(VariablesManager variablesManager, Layer[] inputLayers) {
        super(variablesManager, inputLayers, inputLayers[0].outputSize());
        for (Layer layer : inputLayers)
            if (layer.outputSize() != outputSize)
                throw new IllegalArgumentException("All input layers to an elementwise-multiplication layer " +
                                                           "must have the same output size.");
    }

    @Override
    Vector computeValue(NetworkState state) {
        Vector value = Vectors.ones(outputSize);
        for (Layer layer : inputLayers)
            value.multElementwiseInPlace(layer.value(state));
        return value;
    }

    @Override
    Matrix localGradient(NetworkState state, Variable variable) {
        if (variable.equals(outputVariable))
            return Matrix.identity(outputSize);
        else if (!ArrayUtilities.contains(inputVariables, variable))
            return Matrix.zeros(outputSize, variable.size());
        else {
            Vector gradientDiagonal = Vectors.ones(outputSize);
            for (Layer inputLayer : inputLayers)
                if (!inputLayer.outputVariable().equals(variable))
                    gradientDiagonal.multElementwiseInPlace(inputLayer.value(state));
            return Matrix.diagonal(gradientDiagonal.getDenseArray());
        }
    }
}
