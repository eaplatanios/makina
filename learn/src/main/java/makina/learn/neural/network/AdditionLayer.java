package makina.learn.neural.network;

import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.utilities.ArrayUtilities;

/**
 * @author Emmanouil Antonios Platanios
 */
class AdditionLayer extends MultiInputLayer {
    AdditionLayer(VariablesManager variablesManager, Layer[] inputLayers) {
        super(variablesManager, inputLayers, inputLayers[0].outputSize());
        for (Layer layer : inputLayers)
            if (layer.outputSize() != outputSize)
                throw new IllegalArgumentException("All input layers to an addition layer " +
                                                           "must have the same output size.");
    }

    @Override
    Vector computeValue(NetworkState state) {
        Vector value = Vectors.build(outputSize, state.vectorType());
        for (Layer layer : inputLayers)
            value.addInPlace(layer.value(state));
        return value;
    }

    @Override
    Matrix localGradient(NetworkState state, Variable variable) {
        if (ArrayUtilities.contains(inputVariables, variable) || variable.equals(outputVariable))
            return Matrix.identity(outputSize);
        else
            return Matrix.zeros(outputSize, variable.size());
    }
}
