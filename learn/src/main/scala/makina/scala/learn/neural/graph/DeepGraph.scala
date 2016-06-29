package makina.scala.learn.neural.graph

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import makina.scala.learn.neural.graph.DeepGraph.VertexContent
import makina.scala.learn.neural.network.{SigmoidActivation, SoftmaxActivation}
import makina.scala.optimization._
//import breeze.optimize.{DiffFunction, LBFGS}
import com.typesafe.scalalogging.Logger
import makina.scala.learn.graph.{Graph, Vertex}
import makina.scala.learn.neural.graph.DeepGraph._
import makina.scala.learn.neural.network.SoftmaxActivation
import org.slf4j.LoggerFactory

import scala.util.Random

/**
  * @author Emmanouil Antonios Platanios
  */
object DeepGraph {
  val logger = Logger(LoggerFactory.getLogger("Deep Graph"))
  val random = Random

  private val solver = LBFGSQuasiNewtonSolver(
    numberOfIterations = 1000,
    numberOfFunctionEvaluations = 10000,
    checkForObjectiveConvergence = true,
    objectiveChangeTolerance = 1e-6,
    checkForGradientConvergence = true,
    gradientTolerance = 1e-6,
    logging = 5,
    lineSearch = StrongWolfeInterpolationLineSearch(
      1.0, ConstantLineSearchInitialization(1.0), 1e-4, 0.9, 1000
    )
//    lineSearch = NoLineSearch(1.0)
//    lineSearch = BacktrackingLineSearch(contraptionFactor = 0.5, c = 0.5)
  )

  case class VertexContent(id: Int,
                           var step: Int = 0,
                           allFeatures: Array[DenseVector[Double]] = null,
                           var featuresGradient: DenseVector[Double] = null,
                           private var _incomingFeaturesSum: Array[DenseVector[Double]] = null,
                           private var _outgoingFeaturesSum: Array[DenseVector[Double]] = null) {
    if (_incomingFeaturesSum == null && allFeatures != null)
      _incomingFeaturesSum = new Array[DenseVector[Double]](allFeatures.length)
    if (_outgoingFeaturesSum == null && allFeatures != null)
      _outgoingFeaturesSum = new Array[DenseVector[Double]](allFeatures.length)

    def features = allFeatures(step)
    def incomingFeaturesSum = _incomingFeaturesSum
    def incomingFeaturesSum(sum: DenseVector[Double], step: Int) = _incomingFeaturesSum(step) = sum
    def outgoingFeaturesSum = _outgoingFeaturesSum
    def outgoingFeaturesSum(sum: DenseVector[Double], step: Int) = _outgoingFeaturesSum(step) = sum
  }

  abstract class UpdateFunction[E >: Null <: AnyRef] {
    def parametersSize: Int
    def initialParameters: DenseVector[Double]
    def value(parameters: DenseVector[Double], vertex: Vertex[VertexContent, E], step: Int): DenseVector[Double]
    def gradient(parameters: DenseVector[Double], vertex: Vertex[VertexContent, E], step: Int): DenseMatrix[Double]
    def featuresGradient(parameters: DenseVector[Double],
                         vertex: Vertex[VertexContent, E],
                         differentiatingVertex: Vertex[VertexContent, E],
                         step: Int): DenseMatrix[Double]
  }

  abstract class OutputFunction {
    def parametersSize: Int
    def initialParameters: DenseVector[Double]
    def value(features: DenseVector[Double], parameters: DenseVector[Double]): DenseVector[Double]
    def featuresGradient(features: DenseVector[Double], parameters: DenseVector[Double]): DenseMatrix[Double]
    def parametersGradient(features: DenseVector[Double], parameters: DenseVector[Double]): DenseMatrix[Double]
  }

  case class VertexClassificationOutputFunction(featuresSize: Int, outputsSize: Int) extends OutputFunction {
    override def parametersSize = (featuresSize + 1) * outputsSize

    override def initialParameters: DenseVector[Double] = {
      val initialParameters = DenseVector.zeros[Double](parametersSize)
      var index = 0
      for (i <- 0 until outputsSize; j <- 0 until featuresSize) {
        initialParameters(index) = (random.nextDouble() - 0.5) * 2.0 / Math.sqrt(outputsSize)
        index += 1
      }
      initialParameters
    }

    override def value(features: DenseVector[Double], parameters: DenseVector[Double]): DenseVector[Double] = {
      val value = DenseVector.zeros[Double](outputsSize)
      var index = 0
      for (i <- 0 until outputsSize; j <- 0 until featuresSize) {
        value(i) += parameters(index) * features(j)
        index += 1
      }
      for (i <- 0 until outputsSize) {
        value(i) += parameters(index)
        index += 1
      }
      if (outputsSize == 1)
        SigmoidActivation.value(value)
      else
        SoftmaxActivation.value(value)
    }

    override def featuresGradient(features: DenseVector[Double],
                                  parameters: DenseVector[Double]): DenseMatrix[Double] = {
      val value = DenseVector.zeros[Double](outputsSize)
      val gradient = DenseMatrix.zeros[Double](outputsSize, featuresSize)
      var index = 0
      for (i <- 0 until outputsSize; j <- 0 until featuresSize) {
        value(i) += parameters(index) * features(j)
        gradient(i, j) = parameters(index)
        index += 1
      }
      for (i <- 0 until outputsSize) {
        value(i) += parameters(index)
        index += 1
      }
      if (outputsSize == 1)
        SigmoidActivation.gradient(value) * gradient
      else
        SoftmaxActivation.gradient(value) * gradient
    }

    override def parametersGradient(features: DenseVector[Double],
                                    parameters: DenseVector[Double]): DenseMatrix[Double] = {
      val value = DenseVector.zeros[Double](outputsSize)
      val gradient = DenseMatrix.zeros[Double](outputsSize, parametersSize)
      var index = 0
      for (i <- 0 until outputsSize; j <- 0 until featuresSize) {
        value(i) += parameters(index) * features(j)
        gradient(i, index) = features(j)
        index += 1
      }
      for (i <- 0 until outputsSize) {
        value(i) += parameters(index)
        gradient(i, index) = 1.0
        index += 1
      }
      if (outputsSize == 1)
        SigmoidActivation.gradient(value) * gradient
      else
        SoftmaxActivation.gradient(value) * gradient
    }
  }

  abstract class LossFunction {
    def value(networkOutput: DenseVector[Double], trueOutput: DenseVector[Double]): Double
    def gradient(networkOutput: DenseVector[Double], trueOutput: DenseVector[Double]): DenseVector[Double]
  }

  case class CrossEntropyLossFunction(outputsSize: Int) extends LossFunction {
    override def value(networkOutput: DenseVector[Double], trueOutput: DenseVector[Double]): Double = {
      var value = 0.0
      if (outputsSize == 1) {
        trueOutput.foreachPair((i, v) => {
          if (v >= 0.5)
            value -= Math.log(networkOutput(i))
          else
            value -= Math.log(1 - networkOutput(i))
        })
      } else {
        val output = networkOutput(trueOutput(0).asInstanceOf[Int])
        if (output > 0)
          value -= Math.log(output)
        else
          value += Double.MaxValue
      }
      value
    }

    override def gradient(networkOutput: DenseVector[Double], trueOutput: DenseVector[Double]): DenseVector[Double] = {
      val gradient = DenseVector.zeros[Double](networkOutput.length)
      if (outputsSize == 1) {
        trueOutput.foreachPair((i, v) => {
          val output = networkOutput(i)
          if (v >= 0.5 && output == 0.0) // TODO: Fix this.
            gradient(i) = -Double.MaxValue
          else if (v < 0.5 && output == 1.0)
            gradient(i) = Double.MaxValue
          else
            gradient(i) = (output - v) / (output * (1 - output))
        })
      } else {
        val index = trueOutput(0).asInstanceOf[Int]
        val output = networkOutput(index)
        if (output > 0)
          gradient(index) = -1.0 / output
        else
          gradient(index) = -Double.MaxValue
      }
      gradient
    }
  }
}

case class DeepGraph[E >: Null <: AnyRef](featuresSize: Int,
                                          outputsSize: Int,
                                          numberOfSteps: Int,
                                          graph: Graph[VertexContent, E],
                                          updateFunction: UpdateFunction[E],
                                          outputFunction: OutputFunction,
                                          lossFunction: LossFunction) {
  private val objectiveFunction = ObjectiveFunction(parametersSize)
  private var needsForwardPass = true
  private var needsBackwardPass = true
  private var isTraining = false
  private var trainingData: Map[Int, DenseVector[Double]] = null
  private var loss: Double = 0.0
  private var _updateFunctionParameters = DenseVector.zeros[Double](updateFunction.parametersSize)
  private var _outputFunctionParameters = DenseVector.zeros[Double](outputFunction.parametersSize)
  private var _updateFunctionParametersGradient = DenseVector.zeros[Double](updateFunction.parametersSize)
  private var _outputFunctionParametersGradient = DenseVector.zeros[Double](outputFunction.parametersSize)
  reset()

  def updateFunctionParameters(): DenseVector[Double] = _updateFunctionParameters
  def updateFunctionParameters(updateFunctionParameters: DenseVector[Double]) = {
    _updateFunctionParameters = updateFunctionParameters
    if (!needsForwardPass || !needsBackwardPass)
      reset()
  }

  def outputFunctionParameters(): DenseVector[Double] = _outputFunctionParameters
  def outputFunctionParameters(outputFunctionParameters: DenseVector[Double]) = {
    _outputFunctionParameters = outputFunctionParameters
    if (!needsForwardPass || !needsBackwardPass)
      reset()
  }

  def parametersSize = updateFunction.parametersSize + outputFunction.parametersSize

  def output(vertex: Vertex[VertexContent, E]): DenseVector[Double] = {
    forwardPass()
    outputFunction.value(vertex.content.features, _outputFunctionParameters)
  }

  def lossValue(): Double = {
    forwardPass()
    loss
  }

  def checkGradient(tolerance: Double): Boolean = {
    randomize()
    val point = DenseVector.rand[Double](parametersSize)
    val gradientApproximation = CentralDifferenceGradientApproximation.gradient(objectiveFunction, point)
    val actualGradient = objectiveFunction.gradient(point)
    val isCorrect = norm(gradientApproximation - actualGradient) < tolerance
    reset()
    isCorrect
  }

  def train(trainingData: Map[Int, DenseVector[Double]]) = {
    this.trainingData = trainingData
    needsForwardPass = true
    needsBackwardPass = true
    isTraining = true
    if (!checkGradient(1e-5))
      logger.warn("The gradient is not the same as the one obtained by the method of finite differences.")
    val initialPoint = DenseVector.zeros[Double](updateFunction.parametersSize + outputFunction.parametersSize)
    val middleIndex = updateFunction.parametersSize
    initialPoint(0 until middleIndex) := updateFunction.initialParameters
    initialPoint(middleIndex until (outputFunction.parametersSize + middleIndex)) := outputFunction.initialParameters
//    val objective = new DiffFunction[DenseVector[Double]] {
//      def calculate(x: DenseVector[Double]) = (objectiveFunction.value(x), objectiveFunction.gradient(x))
//    }
//    val solver = new LBFGS[DenseVector[Double]](maxIter = 1000, m = 10, tolerance = 1e-6)
//    val solution: DenseVector[Double] = solver.minimize(objective, initialPoint)
    val solution = solver.minimize(objectiveFunction, initialPoint)
    updateFunctionParameters(solution(0 until middleIndex))
    outputFunctionParameters(solution(middleIndex until (outputFunction.parametersSize + middleIndex)))
    isTraining = false
  }

  def updateFunctionParametersGradient(): DenseVector[Double] = {
    forwardPass()
    backwardPass()
    _updateFunctionParametersGradient
  }

  def outputFunctionParametersGradient(): DenseVector[Double] = {
    forwardPass()
    backwardPass()
    _outputFunctionParametersGradient
  }

  def reset() = {
    graph.computeVerticesUpdatedContent(resetVertexUpdateFunction)
    graph.updateVerticesContent()
    graph.computeVerticesUpdatedContent(neighborsSumVertexUpdateFunction)
    needsForwardPass = true
    needsBackwardPass = true
  }

  def randomize() = {
    graph.computeVerticesUpdatedContent(randomizeVertexUpdateFunction)
    graph.updateVerticesContent()
    graph.computeVerticesUpdatedContent(neighborsSumVertexUpdateFunction)
    needsForwardPass = true
    needsBackwardPass = true
  }

  def forwardPass() = {
    if (needsForwardPass) {
      loss = 0
      for (step <- 0 until numberOfSteps) {
        graph.computeVerticesUpdatedContent(forwardVertexUpdateFunction)
//        graph.updateVerticesContent()
        graph.computeVerticesUpdatedContent(neighborsSumVertexUpdateFunction)
      }
      needsForwardPass = false
    }
  }

  def backwardPass() = {
    if (needsBackwardPass) {
      _updateFunctionParametersGradient = DenseVector.zeros[Double](updateFunction.parametersSize)
      _outputFunctionParametersGradient = DenseVector.zeros[Double](outputFunction.parametersSize)
      for (step <- numberOfSteps until 0 by -1) {
        graph.computeVerticesUpdatedContent(backwardVertexUpdateFunction)
//        graph.updateVerticesContent()
      }
      needsBackwardPass = false
    }
  }

  private def resetVertexUpdateFunction(vertex: Vertex[VertexContent, E]): VertexContent = {
    val features = Array.ofDim[DenseVector[Double]](numberOfSteps + 1)
    features(0) = DenseVector.zeros[Double](featuresSize)
    VertexContent(vertex.content.id, 0, features)
  }

  private def randomizeVertexUpdateFunction(vertex: Vertex[VertexContent, E]): VertexContent = {
    val features = Array.ofDim[DenseVector[Double]](numberOfSteps + 1)
    features(0) = DenseVector.rand[Double](featuresSize)
    VertexContent(vertex.content.id, 0, features)
  }

  private def neighborsSumVertexUpdateFunction(vertex: Vertex[VertexContent, E]): VertexContent = {
    vertex.content.incomingFeaturesSum(
      vertex.incomingEdges.map(_.source.content.features).foldLeft(DenseVector.zeros[Double](featuresSize))(_+_),
      vertex.content.step
    )
    vertex.content.outgoingFeaturesSum(
      vertex.outgoingEdges.map(_.destination.content.features).foldLeft(DenseVector.zeros[Double](featuresSize))(_+_),
      vertex.content.step
    )
    null
  }

  private def forwardVertexUpdateFunction(vertex: Vertex[VertexContent, E]): VertexContent = {
    vertex.content.allFeatures(vertex.content.step + 1) =
      updateFunction.value(_updateFunctionParameters, vertex, vertex.content.step)
    if (isTraining && vertex.content.step == numberOfSteps - 1) {
      trainingData.get(vertex.content.id).foreach(trueOutput => {
        this.synchronized {
          loss += lossFunction.value(outputFunction.value(vertex.content.allFeatures(vertex.content.step + 1),
                                                          _outputFunctionParameters), trueOutput)
        }
      })
    }
    vertex.content.step += 1
    null
//    vertex.content.copy(step = vertex.content.step + 1)
//    VertexContent(vertex.content.id,
//                  vertex.content.step + 1,
//                  vertex.content.allFeatures,
//                  null,
//                  vertex.content.incomingFeaturesSum,
//                  vertex.content.outgoingFeaturesSum)
  }

  private def backwardVertexUpdateFunction(vertex: Vertex[VertexContent, E]): VertexContent = {
    var featuresGradient: DenseVector[Double] = DenseVector.zeros[Double](featuresSize)
    if (vertex.content.step == numberOfSteps) {
      trainingData.get(vertex.content.id).foreach(trueOutput => {
        val lossGradient = lossFunction.gradient(outputFunction.value(vertex.content.features, _outputFunctionParameters), trueOutput)
        this.synchronized(_outputFunctionParametersGradient +=
                          outputFunction.parametersGradient(vertex.content.features, _outputFunctionParameters).t * lossGradient)
        featuresGradient = outputFunction.featuresGradient(vertex.content.features, _outputFunctionParameters).t * lossGradient
      })
    } else {
      featuresGradient = updateFunction.featuresGradient(_updateFunctionParameters, vertex, vertex, vertex.content.step).t * vertex.content.featuresGradient
      vertex.incomingEdges.foreach(edge => {
        featuresGradient += updateFunction.featuresGradient(_updateFunctionParameters, edge.source, vertex, vertex.content.step).t *
                            edge.source.content.featuresGradient
      })
      vertex.outgoingEdges.foreach(edge => {
        featuresGradient += updateFunction.featuresGradient(_updateFunctionParameters, edge.destination, vertex, vertex.content.step).t *
                            edge.destination.content.featuresGradient
      })
    }
    this.synchronized(_updateFunctionParametersGradient +=
                      updateFunction.gradient(_updateFunctionParameters, vertex, vertex.content.step - 1).t * featuresGradient)
    vertex.content.step -= 1
    vertex.content.featuresGradient = featuresGradient
    null
//    vertex.content.copy(step = vertex.content.step - 1, featuresGradient = featuresGradient)
//    VertexContent(vertex.content.id,
//                  vertex.content.step - 1,
//                  vertex.content.allFeatures,
//                  featuresGradient,
//                  vertex.content.incomingFeaturesSum,
//                  vertex.content.outgoingFeaturesSum)
  }

  private case class ObjectiveFunction(inputSize: Int) extends Function {
    private var oldPoint: DenseVector[Double] = DenseVector.zeros[Double](inputSize)

    private def checkPoint(point: DenseVector[Double]) {
      if (norm(point - oldPoint) > 1e-10) {
        val middleIndex = updateFunction.parametersSize
        updateFunctionParameters(point(0 until middleIndex))
        outputFunctionParameters(point(middleIndex until (outputFunction.parametersSize + middleIndex)))
        oldPoint = point
      }
    }

    override protected def computeValue(point: DenseVector[Double]): Double = {
      checkPoint(point)
      lossValue()
    }

    override protected def computeGradient(point: DenseVector[Double]): DenseVector[Double] = {
      checkPoint(point)
      val gradient = DenseVector.zeros[Double](outputFunction.parametersSize + updateFunction.parametersSize)
      val middleIndex = updateFunction.parametersSize
      gradient(0 until middleIndex) := updateFunctionParametersGradient()
      gradient(middleIndex until (outputFunction.parametersSize + middleIndex)) := outputFunctionParametersGradient()
      gradient
    }
  }
}
