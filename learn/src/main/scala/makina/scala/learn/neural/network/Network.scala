package makina.scala.learn.neural.network

import breeze.linalg.{DenseMatrix, DenseVector}
import makina.scala.optimization.Function

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class Network private (private val builder: Network.Builder) {
  private val variablesManager = builder.variablesManager
  private val state = NetworkState(variablesManager)
  private val layers = builder.layers.toArray
  private val inputLayers = builder.inputLayers.toArray
  private val outputLayer = builder.outputLayer
  private val inputSize = inputLayers.length
  private val outputSize = outputLayer.outputSize
  private val variables = builder.variables.toArray
//  private val parameters = builder.parameters.toArray

//  def parameterNames(): Array[String] = parameters.map(_.name)
  def variableNames(): Array[String] = variables.map(_.name)

  def set(variable: Variable, value: DenseVector[Double]) = state.set(variable, value)
  def set(id: Int, value: DenseVector[Double]) = state.set(id, value)
  def set(name: String, value: DenseVector[Double]) = state.set(name, value)

  def get(variable: Variable) = state.get(variable)
  def get(id: Int) = state.get(id)
  def get(name: String) = state.get(name)

  def value: DenseVector[Double] = outputLayer.value(state)

  def recursiveGradient(variable: Variable): DenseMatrix[Double] = outputLayer.recursiveGradient(state, variable)
  def recursiveGradient(id: Int): DenseMatrix[Double] = recursiveGradient(variablesManager.get(id))
  def recursiveGradient(name: String): DenseMatrix[Double] = recursiveGradient(variablesManager.get(name))
  def recursiveGradient(variables: Array[Variable]): Array[DenseMatrix[Double]] = variables.map(recursiveGradient)
  def recursiveGradient(ids: Array[Int]): Array[DenseMatrix[Double]] = ids.map(recursiveGradient)
  def recursiveGradient(names: Array[String]): Array[DenseMatrix[Double]] = names.map(recursiveGradient)

  def gradient(variable: Variable): DenseMatrix[Double] = {
    layers.foreach(_.resetForwardGradient())
    outputLayer.backPropagateGradient(state, outputLayer.localGradient(state, outputLayer.outputVariable))
    val gradient = DenseMatrix.zeros[Double](outputSize, variable.size)
    var currentBackPropagationIndex: Int = layers.length - 1
    while (currentBackPropagationIndex >= 0) {
      var layerIndex = currentBackPropagationIndex
      while (layerIndex > -1) {
        val layerGradient = layers(layerIndex).gradient(state, variable)
        if (layerGradient != null) {
          gradient += layerGradient
          val swapLayer = layers(layerIndex)
          layers(layerIndex) = layers(currentBackPropagationIndex)
          layers(currentBackPropagationIndex) = swapLayer
          layerIndex = currentBackPropagationIndex
          currentBackPropagationIndex -= 1
        }
        layerIndex -= 1
      }
    }
    gradient
  }
  def gradient(id: Int): DenseMatrix[Double] = gradient(variablesManager.get(id))
  def gradient(name: String): DenseMatrix[Double] = gradient(variablesManager.get(name))
  def gradient(variables: Array[Variable]): Array[DenseMatrix[Double]] = {
    layers.foreach(_.resetForwardGradient())
    outputLayer.backPropagateGradient(state, outputLayer.localGradient(state, outputLayer.outputVariable))
    val gradients = Array.ofDim[DenseMatrix[Double]](variables.length)
    for (i <- gradients.indices)
      gradients(i) = DenseMatrix.zeros[Double](outputSize, variables(i).size)
    var currentBackPropagationIndex: Int = layers.length - 1
    while (currentBackPropagationIndex >= 0) {
      var layerIndex = currentBackPropagationIndex
      while (layerIndex > -1) {
        val layerGradients = layers(layerIndex).gradient(state, variables)
        if (layerGradients != null) {
          for (i <- gradients.indices)
            gradients(i) += layerGradients(i)
          val swapLayer = layers(layerIndex)
          layers(layerIndex) = layers(currentBackPropagationIndex)
          layers(currentBackPropagationIndex) = swapLayer
          layerIndex = currentBackPropagationIndex
          currentBackPropagationIndex -= 1
        }
        layerIndex -= 1
      }
    }
    gradients
  }
  def gradient(ids: Array[Int]): Array[DenseMatrix[Double]] = gradient(ids.map(variablesManager.get))
  def gradient(names: Array[String]): Array[DenseMatrix[Double]] = gradient(names.map(variablesManager.get))

  def objectiveFunction(lossFunction: LossFunction, variables: Array[Variable]) =
    ObjectiveFunction(lossFunction, variables)
  def objectiveFunction(lossFunction: LossFunction, ids: Array[Int]) =
    ObjectiveFunction(lossFunction, ids.map(variablesManager.get))
  def objectiveFunction(lossFunction: LossFunction, names: Array[String]) =
    ObjectiveFunction(lossFunction, names.map(variablesManager.get))

  class ObjectiveFunction private (private val lossFunction: LossFunction,
                                   private val variables: Array[Variable]) extends Function {
    if (lossFunction.inputSize != outputSize)
      throw new IllegalArgumentException("Input size of loss function not equal to output size of network.")

    override protected def computeValue(point: DenseVector[Double]): Double = {
      var index: Int = 0
      for (variable <- variables) {
        set(variable, point(index until index + variable.size))
        index += variable.size
      }
      lossFunction.value(Network.this.value.toDenseVector)
    }

    override protected def computeGradient(point: DenseVector[Double]): DenseVector[Double] = {
      var index: Int = 0
      for (variable <- variables) {
        set(variable, point(index until index + variable.size))
        index += variable.size
      }
      val lossFunctionGradient = lossFunction.gradient(Network.this.value.toDenseVector)
      val gradient = DenseVector.zeros[Double](point.size)
      index = 0
      for (networkGradient <- Network.this.gradient(variables))
        gradient(index until index + networkGradient.cols) := networkGradient.t * lossFunctionGradient
      gradient
    }
  }

  object ObjectiveFunction {
    def apply(lossFunction: LossFunction, variables: Array[Variable]) = new ObjectiveFunction(lossFunction, variables)
  }
}

object Network {
  class Builder {
    val variablesManager = VariablesManager()
    val layersManager = LayersManager()
    val layers = mutable.ArrayBuffer[Layer]()
    val inputLayers = mutable.ArrayBuffer[Layer]()
    val variables = mutable.Set[Variable]()
//  val parameters = mutable.Set[Variable]()

    var outputLayer: Layer = null

    private def addLayer(layer: Layer, isInput: Boolean = false, isOutput: Boolean = false): Layer = {
      if (layers contains layer)
        throw new IllegalArgumentException("The provided layer has already been added to this network builder.")
      if (isOutput && outputLayer != null)
        throw new IllegalArgumentException("There can only be one output layer for each network.")
      val index = layers.indexWhere(
        _.inputLayers contains layer)       // Make sure the added layers are sorted such that layers that are used
      if (index > -1) layers.insert(index, layer) else layers += layer  // as inputs for other layers, come before them.
      variables ++= layer.inputVariables
      variables += layer.outputVariable
      if (isInput) inputLayers += layer
      if (isOutput) outputLayer = layer
      layersManager.add(layer)
    }

    def addConstantLayer(value: DenseVector[Double]): Layer =
      addLayer(ConstantLayer(variablesManager, value))

    def addInputLayer(size: Int): Layer =
      addLayer(InputLayer(variablesManager, size), isInput = true)

    def addInputLayer(size: Int, name: String): Layer =
      addLayer(InputLayer(variablesManager, size, name), isInput = true)

    def addSigmoidLayer(inputLayer: Layer, isOutput: Boolean = false): Layer =
      addLayer(SigmoidLayer(variablesManager, inputLayer), isOutput = isOutput)

    def addTanhLayer(inputLayer: Layer, isOutput: Boolean = false): Layer =
      addLayer(TanhLayer(variablesManager, inputLayer), isOutput = isOutput)

    def addRectifiedLinearLayer(inputLayer: Layer, threshold: Double = 0.0, isOutput: Boolean = false): Layer =
      addLayer(RectifiedLinearLayer(variablesManager, inputLayer, threshold), isOutput = isOutput)

    def addLeakyRectifiedLinearLayer(inputLayer: Layer,
                                     threshold: Double = 0.0,
                                     alpha: Double = 0.01,
                                     isOutput: Boolean = false): Layer =
      addLayer(LeakyRectifiedLinearLayer(variablesManager, inputLayer, threshold, alpha), isOutput = isOutput)

    def addAdditionLayer(inputLayers: Array[Layer], isOutput: Boolean = false) =
      addLayer(AdditionLayer(variablesManager, inputLayers), isOutput = isOutput)

    def addSubtractionLayer(inputLayer1: Layer, inputLayer2: Layer, isOutput: Boolean = false) =
      addLayer(SubtractionLayer(variablesManager, Array(inputLayer1, inputLayer2)), isOutput = isOutput)

    def addElementwiseMultiplicationLayer(inputLayers: Array[Layer], isOutput: Boolean = false) =
      addLayer(ElementwiseMultiplicationLayer(variablesManager, inputLayers), isOutput = isOutput)

    def addFullyConnectedLayer(inputLayer: Layer,
                               numberOfHiddenUnits: Int,
                               useBias: Boolean = true,
                               isOutput: Boolean = false,
                               weightsVariableName: String = null,
                               biasVariableName: String = null): Layer =
      addLayer(FullyConnectedLayer(variablesManager,
                                   inputLayer,
                                   numberOfHiddenUnits,
                                   useBias,
                                   weightsVariableName,
                                   biasVariableName), isOutput = isOutput)

    def build(): Network = {
      if (outputLayer == null)
        throw new IllegalStateException("A network cannot be built without an output layer.")
      new Network(this)
    }
  }

  object Builder {
    def apply() = new Builder
  }
}
