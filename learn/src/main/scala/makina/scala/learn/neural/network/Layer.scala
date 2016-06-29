package makina.scala.learn.neural.network

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import makina.scala.learn.neural.network
import makina.scala.learn.neural.network.MatrixVariable

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Layer {
  protected val variablesManager: VariablesManager
  val inputLayers: Array[Layer]
  val inputVariables: Array[Variable]
  val outputSize: Int
  val outputVariable: Variable

  private var _outputLayers = Array[Layer]()
  private var _numberOfOutputLayers = 0
  private var _forwardGradient: DenseMatrix[Double] = null
  private var _numberOfForwardGradientsReceived = 0

  def outputLayers: Array[Layer] = _outputLayers

  def addOutputLayer(layer: Layer): Int = {
    _outputLayers = outputLayers :+ layer
    _numberOfOutputLayers += 1
    _numberOfOutputLayers
  }

  def resetForwardGradient() = {
    _forwardGradient = null
    _numberOfForwardGradientsReceived = 0
  }

  def parameters: Array[Variable] = Array[Variable]()

  def value(state: NetworkState): DenseVector[Double] = {
    val value = computeValue(state)
    state.set(outputVariable, value)
    value
  }

  def computeValue(state: NetworkState): DenseVector[Double]
  def localGradient(state: NetworkState, variable: Variable): DenseMatrix[Double]

  def recursiveGradient(state: NetworkState, variable: Variable) : DenseMatrix[Double] = {
    val gradient = localGradient(state, variable)
    for (layer <- inputLayers)
      if (variable != layer.outputVariable)
        gradient += localGradient(state, layer.outputVariable) * layer.recursiveGradient(state, variable)
    gradient
  }

  def recursiveGradient(state: NetworkState, variables: Variable*): Array[DenseMatrix[Double]] = {
    val gradient = Array.ofDim[DenseMatrix[Double]](variables.length)
    for (i <- 0 until variables.length)
      gradient(i) = recursiveGradient(state, variables(i))
    gradient
  }

  def gradient(state: NetworkState, variable: Variable): DenseMatrix[Double] = {
    if (_numberOfForwardGradientsReceived == _numberOfOutputLayers || _numberOfOutputLayers == 0)
      _forwardGradient * localGradient(state, variable)
    else
      null
  }

  def gradient(state: NetworkState, variables: Array[Variable]): Array[DenseMatrix[Double]] = {
    if (_numberOfForwardGradientsReceived == _numberOfOutputLayers || _numberOfOutputLayers == 0) {
      val gradient = Array.ofDim[DenseMatrix[Double]](variables.length)
      for (i <- variables.indices)
        gradient(i) = _forwardGradient * localGradient(state, variables(i))
      gradient
    } else {
      null
    }
  }

  def backPropagateGradient(state: NetworkState, forwardGradient: DenseMatrix[Double]): Unit = {
    if (_numberOfForwardGradientsReceived == 0 || _numberOfOutputLayers == 0)
      _forwardGradient = forwardGradient
    else if (_numberOfForwardGradientsReceived < _numberOfOutputLayers)
      _forwardGradient += forwardGradient
    _numberOfForwardGradientsReceived += 1
    if (_numberOfForwardGradientsReceived == _numberOfOutputLayers || _numberOfOutputLayers == 0) {
      for (layer <- inputLayers)
        layer.backPropagateGradient(state, _forwardGradient * localGradient(state, layer.outputVariable))
    }
  }
}

case class ConstantLayer(variablesManager: VariablesManager, value: DenseVector[Double]) extends Layer {
  override val inputLayers: Array[Layer] = Array[Layer]()
  override val inputVariables: Array[Variable] = Array[Variable]()
  override val outputSize: Int = value.size
  override val outputVariable: Variable = variablesManager.constantVariable(value)

  override def computeValue(state: NetworkState): DenseVector[Double] = value
  override def localGradient(state: NetworkState, variable: Variable): DenseMatrix[Double] = {
    if (variable == outputVariable)
      DenseMatrix.eye[Double](outputSize)
    else
      DenseMatrix.zeros[Double](outputSize, variable.size)
  }

  override def recursiveGradient(state: NetworkState, variable: Variable) : DenseMatrix[Double] =
    DenseMatrix.zeros[Double](outputSize, variable.size)
}

case class InputLayer(variablesManager: VariablesManager, inputSize: Int, name: String = null) extends Layer {
  val inputVariable: Variable =
    if (name == null)
      variablesManager.vectorVariable(inputSize)
    else
      variablesManager.vectorVariable(name, inputSize)
  override val inputLayers: Array[Layer] = Array[Layer]()
  override val inputVariables: Array[Variable] = Array[Variable](inputVariable)
  override val outputSize: Int = inputSize
  override val outputVariable: Variable = variablesManager.layerVariable(this)

  override def computeValue(state: NetworkState): DenseVector[Double] = state.get(inputVariable)
  override def localGradient(state: NetworkState, variable: Variable): DenseMatrix[Double] = {
    if (variable == outputVariable || variable == inputVariable)
      DenseMatrix.eye[Double](outputSize)
    else
      DenseMatrix.zeros[Double](outputSize, variable.size)
  }

  override def recursiveGradient(state: NetworkState, variable: Variable) : DenseMatrix[Double] = {
    if (variable == outputVariable || variable == inputVariable)
      DenseMatrix.eye[Double](inputSize)
    else
      DenseMatrix.zeros[Double](outputSize, variable.size)
  }
}

abstract class SingleInputLayer extends Layer {
  val inputLayer: Layer
  inputLayer.addOutputLayer(this)

  val inputSize = inputLayer.outputSize
  val inputVariable = inputLayer.outputVariable
  override val inputLayers = Array[Layer](inputLayer)
  override val inputVariables = Array[Variable](inputVariable)
  override val outputVariable = variablesManager.layerVariable(this)
}

abstract class ActivationLayer extends SingleInputLayer {
  override val outputSize: Int = inputLayer.outputSize

  override def computeValue(state: NetworkState): DenseVector[Double] = {
    val outputValue = DenseVector.zeros[Double](outputSize)
    inputLayer.value(state).foreachPair((i, v) => outputValue(i) = value(v))
    outputValue
  }

  override def localGradient(state: NetworkState, variable: Variable): DenseMatrix[Double] = {
    variable match {
      case `outputVariable` => DenseMatrix.eye[Double](outputSize)
      case `inputVariable` =>
        val localGradient = DenseMatrix.zeros[Double](outputSize, outputSize)
        inputLayer.value(state).foreachPair((i, v) => localGradient(i, i) = gradient(v))
        localGradient
      case _ => DenseMatrix.zeros[Double](outputSize, variable.size)
    }
  }

  def value(point: Double): Double
  def gradient(point: Double): Double
}

case class SigmoidLayer(variablesManager: VariablesManager, inputLayer: Layer) extends ActivationLayer {
  override def value(point: Double): Double = 1 / (1 + Math.exp(-point))
  override def gradient(point: Double): Double = value(point) * (1 - value(point))
}

case class TanhLayer(variablesManager: VariablesManager, inputLayer: Layer) extends ActivationLayer {
  override def value(point: Double): Double = Math.tanh(point)
  override def gradient(point: Double): Double = 1 - value(point) * value(point)
}

case class RectifiedLinearLayer(variablesManager: VariablesManager,
                                inputLayer: Layer,
                                threshold: Double = 0.0) extends ActivationLayer {
  override def value(point: Double): Double = if (point >= threshold) point else 0.0
  override def gradient(point: Double): Double = if (point >= threshold) 1.0 else 0.0
}

case class LeakyRectifiedLinearLayer(variablesManager: VariablesManager,
                                     inputLayer: Layer,
                                     threshold: Double = 0.0,
                                     alpha: Double = 0.01) extends ActivationLayer {
  override def value(point: Double): Double = if (point >= threshold) point else alpha * point
  override def gradient(point: Double): Double = if (point >= threshold) 1.0 else alpha
}

case class FullyConnectedLayer(variablesManager: VariablesManager,
                               inputLayer: Layer,
                               numberOfHiddenUnits: Int,
                               useBias: Boolean = true,
                               weightsVariableName: String = null,
                               biasVariableName: String = null) extends SingleInputLayer {
  override val outputSize: Int = numberOfHiddenUnits

  val weights: MatrixVariable =
    if (weightsVariableName == null)
      variablesManager.matrixVariable(outputSize, inputSize)
    else
      variablesManager.matrixVariable(weightsVariableName, outputSize, inputSize)

  val bias: VectorVariable =
    if (useBias) {
      if (biasVariableName == null)
        variablesManager.vectorVariable(outputSize)
      else
        variablesManager.vectorVariable(biasVariableName, outputSize)
    } else {
      null
    }

  override def parameters: Array[Variable] =
    if (useBias)
      Array[Variable](weights, bias)
    else
      Array[Variable](weights)

  override def computeValue(state: NetworkState): DenseVector[Double] = {
    val inputValue = inputLayer.value(state)
    val weights = state.get(this.weights)
    val outputValue =
      if (useBias)
        state.get(this.bias).copy
      else
        DenseVector.zeros[Double](outputSize)
    for (i <- 0 until outputSize)
      for (j <- 0 until inputSize)
        outputValue(i) += weights(i + j * outputSize) * inputValue(j)
    outputValue
  }

  override def localGradient(state: NetworkState, variable: Variable): DenseMatrix[Double] = {
    variable match {
      case `outputVariable` => DenseMatrix.eye[Double](outputSize)
      case `inputVariable` => weights.valueInMatrixForm(state)
      case `weights` =>
        val inputValue = inputLayer.value(state)
        val gradient = DenseMatrix.zeros[Double](outputSize, weights.size)
        for (i <- 0 until outputSize)
          for (j <- 0 until inputSize)
            gradient(i, i + j * outputSize) = inputValue(j)
        gradient
      case `bias` if useBias => DenseMatrix.eye[Double](outputSize)
      case _ => DenseMatrix.zeros[Double](outputSize, variable.size)
    }
  }
}

abstract class MultiInputLayer extends Layer {
  override val outputVariable = variablesManager.layerVariable(this)
  private val inputVariablesArrayBuffer = mutable.ArrayBuffer[Variable]()
  for (layer <- inputLayers) {
    inputVariablesArrayBuffer += layer.outputVariable
    layer.addOutputLayer(this)
  }
  override val inputVariables = inputVariablesArrayBuffer.toArray
}

case class AdditionLayer(variablesManager: VariablesManager, inputLayers: Array[Layer]) extends network.MultiInputLayer {
  override val outputSize: Int = inputLayers(0).outputSize
  for (layer <- inputLayers)
    if (layer.outputSize != outputSize)
      throw new IllegalArgumentException("All input layers to an addition layer must have the same output size.")

  override def computeValue(state: NetworkState): DenseVector[Double] = {
    val value = DenseVector.zeros[Double](outputSize)
    for (layer <- inputLayers)
      value += layer.value(state)
    value
  }

  override def localGradient(state: NetworkState, variable: Variable): DenseMatrix[Double] = {
    if (variable.equals(outputVariable) || inputVariables.contains(variable))
      DenseMatrix.eye[Double](outputSize)
    else
      DenseMatrix.zeros[Double](outputSize, variable.size)
  }
}

case class SubtractionLayer(variablesManager: VariablesManager, override val inputLayers: Array[Layer]) extends network.MultiInputLayer {
  def this(variablesManager: VariablesManager, inputLayer1: Layer, inputLayer2: Layer) = this(variablesManager, Array[Layer](inputLayer1, inputLayer2))
  val inputLayer1 = inputLayers(0)
  val inputLayer2 = inputLayers(1)
//  override val inputLayers = Array[Layer](inputLayer1, inputLayer2)
  override val outputSize: Int = inputLayer1.outputSize
  if (inputLayer2.outputSize != outputSize)
    throw new IllegalArgumentException("All input layers to an addition layer must have the same output size.")

  override def computeValue(state: NetworkState): DenseVector[Double] = inputLayer1.value(state) - inputLayer2.value(state)

  override def localGradient(state: NetworkState, variable: Variable): DenseMatrix[Double] = {
    variable match {
      case `outputVariable` => DenseMatrix.eye[Double](outputSize)
      case inputLayer1.outputVariable => DenseMatrix.eye[Double](outputSize)
      case inputLayer2.outputVariable => DenseMatrix.eye[Double](outputSize) * -1.0
      case _ => DenseMatrix.zeros[Double](outputSize, variable.size)
    }
  }
}

case class ElementwiseMultiplicationLayer(variablesManager: VariablesManager, inputLayers: Array[Layer]) extends network.MultiInputLayer {
  override val outputSize: Int = inputLayers(0).outputSize
  for (layer <- inputLayers)
    if (layer.outputSize != outputSize)
      throw new IllegalArgumentException("All input layers to an addition layer must have the same output size.")

  override def computeValue(state: NetworkState): DenseVector[Double] = {
    val value = DenseVector.zeros[Double](outputSize)
    for (layer <- inputLayers)
      value :*= layer.value(state)
    value
  }

  override def localGradient(state: NetworkState, variable: Variable): DenseMatrix[Double] = {
    variable match {
      case `outputVariable` => DenseMatrix.eye[Double](outputSize)
      case  v if inputVariables.contains(v) =>
        val gradientDiagonal = DenseVector.ones[Double](outputSize)
        for (layer <- inputLayers)
          if (v != layer.outputVariable)
            gradientDiagonal :*= layer.value(state)
        diag(gradientDiagonal)
      case _ => DenseMatrix.zeros[Double](outputSize, variable.size)
    }
  }
}
