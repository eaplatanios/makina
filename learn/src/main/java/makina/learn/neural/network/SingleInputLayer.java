package makina.learn.neural.network;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class SingleInputLayer extends Layer {
    final Layer inputLayer;
    final Variable inputVariable;
    final Variable outputVariable;

    SingleInputLayer(VariablesManager variablesManager, Layer inputLayer, int outputSize) {
        super(variablesManager, outputSize);
        this.inputLayer = inputLayer;
        this.inputVariable = inputLayer.outputVariable();
        outputVariable = variablesManager.layerVariable(this);
        inputLayer.addOutputLayer(this);
    }

    @Override
    Layer[] inputLayers() {
        return new Layer[] { inputLayer };
    }

    @Override
    Variable[] inputVariables() {
        return new Variable[] { inputVariable };
    }

    @Override
    Variable outputVariable() {
        return outputVariable;
    }
}
