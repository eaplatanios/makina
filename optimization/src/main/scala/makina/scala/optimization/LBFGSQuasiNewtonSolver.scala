package makina.scala.optimization

import breeze.linalg.{DenseVector, axpy}
import makina.scala.utilities.CircularBuffer

/**
  * @author Emmanouil Antonios Platanios
  */
case class LBFGSQuasiNewtonSolver(
  numberOfIterations: Int = 1000,
  numberOfFunctionEvaluations: Int = 1000000,
  pointChangeTolerance: Double = 1e-10,
  objectiveChangeTolerance: Double = 1e-10,
  gradientTolerance: Double = 1e-6,
  checkForPointConvergence: Boolean = true,
  checkForObjectiveConvergence: Boolean = true,
  checkForGradientConvergence: Boolean = true,
  customConvergenceCriterion: (DenseVector[Double] => Boolean) = point => false,
  logging: Int = 0,
  logObjectiveValue: Boolean = true,
  logGradientNorm: Boolean = true,
  lineSearch: LineSearch = StrongWolfeInterpolationLineSearch(
    1.0, ConserveFirstOrderChangeLineSearchInitialization(), 1e-4, 0.9, 10.0
  ),
  m: Int = 10
) extends LineSearchSolver(numberOfIterations,
                           numberOfFunctionEvaluations,
                           pointChangeTolerance,
                           objectiveChangeTolerance,
                           gradientTolerance,
                           checkForPointConvergence,
                           checkForObjectiveConvergence,
                           checkForGradientConvergence,
                           customConvergenceCriterion,
                           logging,
                           logObjectiveValue,
                           logGradientNorm,
                           lineSearch) {
  private val s = CircularBuffer[DenseVector[Double]](m)
  private val y = CircularBuffer[DenseVector[Double]](m)
  private var initialHessianInverseDiagonal: DenseVector[Double] = null

  override def minimize(objective: Function, initialPoint: DenseVector[Double]): DenseVector[Double] = {
    initialHessianInverseDiagonal = DenseVector.fill[Double](initialPoint.size)(-1.0)
    super.minimize(objective, initialPoint)
  }

  override protected def updatePoint() = {
    point = previousPoint + direction * stepSize
  }

  override protected def updateDirection() = {
    updateStoredVectors()
    if (iteration > 0)
      initialHessianInverseDiagonal = DenseVector.fill[Double](point.size)(-(s(-1).t * y(-1)) / (y(-1).t * y(-1)))
    val min = Math.min(iteration, m)
    val a = Array.ofDim[Double](m)
    val rho = Array.ofDim[Double](m)
    direction = gradient.copy
    for (i <- 0 until min) {
      rho(i) = 1.0 / (y(-i - 1).t * s(-i - 1))
      a(i) = s(-i - 1).t * direction * rho(i)
      axpy(-a(i), y(-i - 1), direction)
    }
    direction :*= initialHessianInverseDiagonal
    for (i <- (m - min) until m)
      axpy(-a(m - i - 1) - y(i).t * direction * rho(m - i - 1), s(i), direction)
  }

  protected def updateStoredVectors() = {
    s.add(point - previousPoint)
    y.add(gradient - previousGradient)
  }
}
