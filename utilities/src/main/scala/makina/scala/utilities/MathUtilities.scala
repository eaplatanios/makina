package makina.scala.utilities

import breeze.linalg._

/**
  * @author Emmanouil Antonios Platanios
  */
object MathUtilities {
  def denseOneHotVector(size: Int, index: Int): Vector[Double] = {
    val vector = DenseVector.zeros[Double](size)
    vector(index) = 1
    vector
  }

  def machineEpsilonDouble: Double = {
    var epsilon: Double = 1
    while (1 + epsilon / 2 > 1.0)
      epsilon /= 2
    epsilon
  }
}
