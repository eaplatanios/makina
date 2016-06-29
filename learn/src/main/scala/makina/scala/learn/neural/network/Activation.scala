package makina.scala.learn.neural.network

import breeze.linalg.{DenseMatrix, DenseVector, diag, sum}
import breeze.numerics.exp

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Activation {
  def value(point: DenseVector[Double]): DenseVector[Double]
  def gradient(point: DenseVector[Double]): DenseMatrix[Double]
}

object SigmoidActivation extends Activation {
  def value(point: Double): Double = 1 / (1 + Math.exp(-point))
  override def value(point: DenseVector[Double]): DenseVector[Double] = point.map(value(_))
  def gradient(point: Double): Double = value(point) * (1 - value(point))
  override def gradient(point: DenseVector[Double]): DenseMatrix[Double] = diag(point.map(gradient(_)))
}

object SoftmaxActivation extends Activation {
  override def value(point: DenseVector[Double]): DenseVector[Double] = {
    val denominator = sum(exp(point))
    point.map(exp(_) / denominator)
  }

  override def gradient(point: DenseVector[Double]): DenseMatrix[Double] = {
    val softmaxValue = value(point)
    val gradient = DenseMatrix.zeros[Double](point.length, point.length)
    for (outputIndex <- 0 until point.length; inputIndex <- 0 until point.length)
      if (outputIndex == inputIndex)
        gradient(outputIndex, inputIndex) = softmaxValue(outputIndex) - softmaxValue(inputIndex)
      else
        gradient(outputIndex, inputIndex) = -softmaxValue(inputIndex)
    gradient
  }
}
