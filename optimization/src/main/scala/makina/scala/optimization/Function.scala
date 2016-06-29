package makina.scala.optimization

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Function {
  private var computeGradientMethodOverriden: Boolean = true
  private var gradientApproximation: GradientApproximation = null

  var numberOfFunctionEvaluations: Int = 0
  var numberOfGradientEvaluations: Int = 0
  var numberOfHessianEvaluations: Int = 0

  /** Computes the function value at a particular point.
    *
    * @param  point The point in which to evaluate the function.
    * @return The value of the function, evaluated at the given point.
    */
  final def value(point: DenseVector[Double]): Double = {
    numberOfFunctionEvaluations += 1
    computeValue(point)
  }

  protected def computeValue(point: DenseVector[Double]): Double

  /** Computes the first derivatives of the function at a particular point.
    *
    * @param  point The point in which to evaluate the derivatives.
    * @return The values of the first derivatives of the function, evaluated at the given point.
    */
  final def gradient(point: DenseVector[Double]): DenseVector[Double] = {
    numberOfGradientEvaluations += 1
    computeGradient(point)
  }

  protected def computeGradient(point: DenseVector[Double]): DenseVector[Double] = {
    if (computeGradientMethodOverriden) {
      computeGradientMethodOverriden = false
      gradientApproximation = CentralDifferenceGradientApproximation
    }
    gradientApproximation.gradient(this, point)
  }

  /** Computes the Hessian of the function at a particular point.
    *
    * @param  point The point in which to evaluate the Hessian.
    * @return The value of the Hessian matrix of the function, evaluated at the given point.
    */
  final def hessian(point: DenseVector[Double]): DenseMatrix[Double] = {
    numberOfHessianEvaluations += 1
    computeHessian(point)
  }

  protected def computeHessian(point: DenseVector[Double]): DenseMatrix[Double] = {
    if (gradientApproximation == null)
      gradientApproximation = CentralDifferenceGradientApproximation
    if (computeGradientMethodOverriden)
      gradientApproximation.hessianGivenGradient(this, point)
    else
      gradientApproximation.hessian(this, point)
  }

  final def setGradientApproximation(gradientApproximation: GradientApproximation): Unit = {
    this.gradientApproximation = gradientApproximation
  }
}

/** Class for a quadratic function of the form \(f(x)=\frac{1}{2}x^TAx-b^Tx\).
  *
  * @author Emmanouil Antonios Platanios
  */
case class QuadraticFunction(A: DenseMatrix[Double], b: DenseVector[Double]) extends Function {
  override protected def computeValue(point: DenseVector[Double]): Double = point.t * A * point * 0.5 - b.t * point
  override protected def computeGradient(point: DenseVector[Double]): DenseVector[Double] = A * point - b
  override protected def computeHessian(point: DenseVector[Double]): DenseMatrix[Double] = A
}
