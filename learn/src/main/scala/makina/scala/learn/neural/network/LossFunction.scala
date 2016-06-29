package makina.scala.learn.neural.network

import breeze.linalg.{DenseVector, norm, sum}
import breeze.numerics.{log, pow}

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class LossFunction {
  val trueOutput: DenseVector[Double]

  val inputSize: Int = trueOutput.size

  def value(networkOutput: DenseVector[Double]): Double
  def gradient(networkOutput: DenseVector[Double]): DenseVector[Double]
}

case class CrossEntropyLossFunction(trueOutput: DenseVector[Double]) extends LossFunction {
  override def value(networkOutput: DenseVector[Double]): Double =
    sum((log(networkOutput) :* trueOutput * -1.0) - log(networkOutput.map(1 - _) :* trueOutput.map(1 - _)))

  override def gradient(networkOutput: DenseVector[Double]): DenseVector[Double] =
    (trueOutput.map(1 - _) :/ networkOutput.map(1 - _)) - (trueOutput :/ networkOutput)
}

case class DifferenceL2NormSquaredLossFunction(trueOutput: DenseVector[Double]) extends LossFunction {
  override def value(networkOutput: DenseVector[Double]): Double = pow(norm(networkOutput - trueOutput), 2)
  override def gradient(networkOutput: DenseVector[Double]): DenseVector[Double] = (networkOutput - trueOutput) * 2.0
}
