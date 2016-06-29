package makina.scala.optimization

import breeze.linalg.{DenseMatrix, DenseVector}
import makina.scala.utilities.MathUtilities._

/**
  * TODO: Jacobian approximation.
  *
  * @author Emmanouil Antonios Platanios
  */
trait GradientApproximation {
  val epsilon: Double

  def gradient(function: Function, point: DenseVector[Double]): DenseVector[Double]
  def hessian(function: Function, point: DenseVector[Double]): DenseMatrix[Double]
  def hessianGivenGradient(function: Function, point: DenseVector[Double]): DenseMatrix[Double]
  def hessianVectorProductGivenGradient(function: Function,
                                        point: DenseVector[Double],
                                        p: DenseVector[Double]) : DenseVector[Double]
}

object ForwardDifferenceGradientApproximation extends GradientApproximation {
  override val epsilon: Double = Math.sqrt(machineEpsilonDouble)

  override def gradient(function: Function, point: DenseVector[Double]): DenseVector[Double] = {
    val currentValue = function.value(point)
    val gradient = DenseVector.zeros[Double](point.size)
    for (i <- 0 until point.size) {
      val ei = denseOneHotVector(point.size, i)
      val forwardValue = function.value(point + ei * epsilon)
      gradient(i) = (forwardValue - currentValue) / epsilon
    }
    gradient
  }

  override def hessian(function: Function, point: DenseVector[Double]): DenseMatrix[Double] = {
    val currentValue = function.value(point)
    val hessian = DenseMatrix.zeros[Double](point.size, point.size)
    for (i <- 0 until point.size) {
      val ei = denseOneHotVector(point.size, i)
      val iValue = function.value(point + ei * epsilon)
      for (j <- i until point.size) {
        val ej = denseOneHotVector(point.size, j)
        val jValue = function.value(point + ej * epsilon)
        val ijValue = function.value(point + (ei + ej) * epsilon)
        val ijEntry = (ijValue - iValue - jValue + currentValue) / Math.pow(epsilon, 2)
        hessian(i, j) = ijEntry
        if (i != j)
          hessian(j, i) = ijEntry
      }
    }
    hessian
  }

  override def hessianGivenGradient(function: Function, point: DenseVector[Double]): DenseMatrix[Double] = {
    val currentGradient = function.gradient(point)
    val hessian = DenseMatrix.zeros[Double](point.size, point.size)
    for (i <- 0 until point.size) {
      val ei = denseOneHotVector(point.size, i)
      val forwardGradient = function.gradient(point + ei * epsilon)
      hessian(i, ::) := (forwardGradient - currentGradient).t / epsilon
    }
    hessian
  }

  override def hessianVectorProductGivenGradient(function: Function,
                                                 point: DenseVector[Double],
                                                 p: DenseVector[Double]): DenseVector[Double] = {
    val currentGradient = function.gradient(point)
    val forwardGradient = function.gradient(point + p * epsilon)
    (forwardGradient - currentGradient) / epsilon
  }
}

object CentralDifferenceGradientApproximation extends GradientApproximation {
  override val epsilon: Double = Math.cbrt(machineEpsilonDouble)

  override def gradient(function: Function, point: DenseVector[Double]): DenseVector[Double] = {
    val gradient = DenseVector.zeros[Double](point.size)
    for (i <- 0 until point.size) {
      val ei = denseOneHotVector(point.size, i)
      val forwardValue = function.value(point + ei * epsilon)
      val backwardValue = function.value(point - ei * epsilon)
      gradient(i) = (forwardValue - backwardValue) / (2 * epsilon)
    }
    gradient
  }

  override def hessian(function: Function, point: DenseVector[Double]): DenseMatrix[Double] = {
    val currentValue = function.value(point)
    val hessian = DenseMatrix.zeros[Double](point.size, point.size)
    for (i <- 0 until point.size) {
      val ei = denseOneHotVector(point.size, i)
      for (j <- i until point.size) {
        if (i != j) {
          val ej = denseOneHotVector(point.size, j)
          val term1 = function.value(point + (ei + ej) * epsilon)
          val term2 = function.value(point + (ei - ej) * epsilon)
          val term3 = function.value(point - (ei - ej) * epsilon)
          val term4 = function.value(point - (ei + ej) * epsilon)
          val ijEntry = (term1 - term2 - term3 + term4) / (4 * Math.pow(epsilon, 2))
          hessian(i, j) = ijEntry
          hessian(j, i) = ijEntry
        } else {
          val term1 = function.value(point + ei * (2 * epsilon))
          val term2 = function.value(point + ei * epsilon)
          val term3 = function.value(point - ei * epsilon)
          val term4 = function.value(point - ei * (2 * epsilon))
          val ijEntry = (-term1 + 16 * term2 - 30 * currentValue + 16 * term3 - term4) / (12 * Math.pow(epsilon, 2))
          hessian(i, j) = ijEntry
        }
      }
    }
    hessian
  }

  override def hessianGivenGradient(function: Function, point: DenseVector[Double]): DenseMatrix[Double] = {
    val hessian = DenseMatrix.zeros[Double](point.size, point.size)
    for (i <- 0 until point.size) {
      val ei = denseOneHotVector(point.size, i)
      val forwardGradient = function.gradient(point + ei * epsilon)
      val backwardGradient = function.gradient(point - ei * epsilon)
      hessian(i, ::) := (forwardGradient - backwardGradient).t / (2 * epsilon)
    }
    hessian
  }

  override def hessianVectorProductGivenGradient(function: Function,
                                                 point: DenseVector[Double],
                                                 p: DenseVector[Double]): DenseVector[Double] = {
    val forwardGradient = function.gradient(point + p * epsilon)
    val backwardGradient = function.gradient(point - p * epsilon)
    (forwardGradient - backwardGradient) / (2 * epsilon)
  }
}
