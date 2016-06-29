package makina.scala.learn.neural

import java.lang.Double

import makina.learn.classification.Utilities
import makina.learn.data.{DataSet, PredictedDataInstance}
import makina.math.matrix.Vector
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class NetworkSpecification extends FlatSpec with ShouldMatchers {
  private val filename: String = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt"
  private val trainingDataSet: DataSet[PredictedDataInstance[Vector, Double]] = Utilities.parseCovTypeDataFromFile(filename, false)

//  val networkBuilder = Network.Builder()
//  val inputLayer = networkBuilder.addInputLayer(featuresSize, "input")
//  val hiddenLayer = networkBuilder.addFullyConnectedLayer(inputLayer, featuresSize, weightsVariableName = "W", biasVariableName = "b")
//  networkBuilder.addSigmoidLayer(hiddenLayer, isOutput = true)
//  val network = networkBuilder.build()
//
//  "Multi-Layer Perceptron (MLP) " should "train" in {
//
//
//
//
//
//  }
//
//  private object MSELossFunction extends LossFunction {
//    override val trueOutput: DenseVector[Double] = _
//
//    override def value(networkOutput: DenseVector[Double]): Double = {
//
//    }
//
//    override def gradient(networkOutput: DenseVector[Double]): DenseVector[Double] = {
//
//    }
//  }
}
