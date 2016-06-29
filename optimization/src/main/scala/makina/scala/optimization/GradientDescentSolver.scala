package makina.scala.optimization

import breeze.linalg.DenseVector

/**
  * @author Emmanouil Antonios Platanios
  */
case class GradientDescentSolver(
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
    1.0, ConserveFirstOrderChangeLineSearchInitialization(), 1e-4, 0.9, 1000
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
  protected def updateDirection() = direction = gradient * -1.0
  protected def updatePoint() = point = previousPoint + direction * stepSize
}
