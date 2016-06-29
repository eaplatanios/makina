package makina.scala.optimization

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * @author Emmanouil Antonios Platanios
  */
case class RosenbrockFunction() extends Function {
  override protected def computeValue(point: DenseVector[Double]): Double =
    100 * Math.pow(point(1) - Math.pow(point(0), 2), 2) + Math.pow(1 - point(0), 2)

  override protected def computeGradient(point: DenseVector[Double]): DenseVector[Double] =
    DenseVector(-400 * (point(1) - Math.pow(point(0), 2)) * point(0) - 2 * (1 - point(0)),
                200 * (point(1) - Math.pow(point(0), 2)))

  override protected def computeHessian(point: DenseVector[Double]): DenseMatrix[Double] =
    DenseMatrix((1200 * Math.pow(point(0), 2) - 400 * point(1) + 2, -400 * point(0)),
                (-400 * point(0), 200.0))
}
