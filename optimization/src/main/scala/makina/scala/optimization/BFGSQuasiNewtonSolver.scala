package makina.scala.optimization

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * @author Emmanouil Antonios Platanios
  */
case class BFGSQuasiNewtonSolver(
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
  )
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
  private var s: DenseMatrix[Double] = null
  private var y: DenseMatrix[Double] = null
  private var H: DenseMatrix[Double] = null
  private var previousH: DenseMatrix[Double] = null

  override def minimize(objective: Function, initialPoint: DenseVector[Double]): DenseVector[Double] = {
    s = DenseMatrix.zeros[Double](initialPoint.size, 1)
    y = DenseMatrix.zeros[Double](initialPoint.size, 1)
    H = DenseMatrix.eye[Double](initialPoint.size)
    previousH = DenseMatrix.eye[Double](initialPoint.size)
    super.minimize(objective, initialPoint)
  }

  override protected def updatePoint() = {
    point = previousPoint + direction * stepSize
  }

  override protected def updateDirection() = {
    updateStoredVectors()
    if (iteration > 0) {
      val sy = (s.t * y).data(0)
      if (iteration == 1)
        previousH := H * (sy / (y.t * y).data(0))
      else
        previousH := H
      val rho = 1 / sy
      H := (DenseMatrix.eye[Double](s.size) - ((s * rho) * y.t)) * previousH *
           (DenseMatrix.eye[Double](s.size) - ((y * rho) * s.t)) + ((s * rho) * s.t)
    }
    direction = H * (gradient * -1.0)
  }

  protected def updateStoredVectors() = {
    s := (point - previousPoint)
    y := (gradient - previousGradient)
  }
}
