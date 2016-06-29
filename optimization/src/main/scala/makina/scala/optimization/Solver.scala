package makina.scala.optimization

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/**
  * @author Emmanouil Antonios Platanios
  */
trait Solver {
  val logger = Logger(LoggerFactory.getLogger("Optimization"))

  def minimize(objective: Function, initialPoint: DenseVector[Double]): DenseVector[Double]
}

/** TODO: Add support for regularization.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class IterativeSolver(numberOfIterations: Int = 1000,
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
                               logGradientNorm: Boolean = true)
  extends Solver {
  protected var objective: Function = null
  protected var point: DenseVector[Double] = null
  protected var value: Double = 0.0
  protected var gradient: DenseVector[Double] = null
  protected var previousPoint: DenseVector[Double] = null
  protected var previousValue: Double = 0.0
  protected var previousGradient: DenseVector[Double] = null
  protected var iteration: Int = 0
  protected var pointChange: Double = Double.MaxValue
  protected var objectiveChange: Double = Double.MaxValue
  protected var gradientNorm: Double = Double.MaxValue
  protected var pointConverged: Boolean = false
  protected var objectiveConverged: Boolean = false
  protected var gradientConverged: Boolean = false

  override def minimize(objective: Function, initialPoint: DenseVector[Double]): DenseVector[Double] = {
    this.objective = objective
    point = initialPoint
    value = objective.value(point)
    previousPoint = point
    previousValue = value
    iteration = 0
    if (logging > 0)
      logger.info("Optimization is starting.")
    while (!checkConvergence() && !customConvergenceCriterion(point)) {
      performIterationUpdates()
      iteration += 1
      if ((logging == 1 && iteration % 1000 == 0) || (logging == 2 && iteration % 100 == 0) ||
          (logging == 3 && iteration % 10 == 0) || logging > 3)
        logIterationState()
    }
    if (logging > 0)
      logTerminationMessage()
    point
  }

  protected def checkConvergence(): Boolean = {
    if (iteration > 0 && iteration >= numberOfIterations)
      true
    else if (iteration > 0 && objective.numberOfFunctionEvaluations >= numberOfFunctionEvaluations)
      true
    else if (iteration > 0) {
      if (checkForPointConvergence) {
        pointChange = norm(point - previousPoint)
        pointConverged = pointChange <= pointChangeTolerance
      }
      if (checkForObjectiveConvergence) {
        objectiveChange = Math.abs((previousValue - value) / previousValue)
        objectiveConverged = objectiveChange <= objectiveChangeTolerance
      }
      if (checkForGradientConvergence) {
//        if (this.isInstanceOf[NonlinearConjugateGradientSolver]) { TODO: Fix this when I implement the solver
//          gradientNorm = Math.abs(currentGradient.max) / (1 + Math.abs(currentObjectiveValue))
//          gradientConverged = gradientNorm <= gradientTolerance
//      } else {
        gradientNorm = norm(gradient)
        gradientConverged = gradientNorm <= gradientTolerance
//      }
      }
      (checkForPointConvergence && pointConverged) ||
      (checkForObjectiveConvergence && objectiveConverged) ||
      (checkForGradientConvergence && gradientConverged)
    } else {
      false
    }
  }

  protected def logIterationState() = {
    if (logObjectiveValue && logGradientNorm)
      logger.info(
        f"Iteration #: $iteration%10d | " +
        f"Func. Eval. #: ${objective.numberOfFunctionEvaluations}%10d | " +
        f"Objective Value: $value%-14.12e | " +
        f"Objective Change: $objectiveChange%-14.12e | " +
        f"Point Change: $pointChange%-14.12e | " +
        f"Gradient Norm: $gradientNorm%-14.12e")
    else if (logObjectiveValue)
      logger.info(
        f"Iteration #: $iteration%10d | " +
        f"Func. Eval. #: ${objective.numberOfFunctionEvaluations}%10d | " +
        f"Objective Value: $value%-14.12e | " +
        f"Objective Change: $objectiveChange%-14.12e | " +
        f"Point Change: $pointChange%-14.12e")
    else if (logGradientNorm)
      logger.info(
        f"Iteration #: $iteration%10d | " +
        f"Func. Eval. #: ${objective.numberOfFunctionEvaluations}%10d | " +
        f"Point Change: $pointChange%-14.12e | " +
        f"Gradient Norm: $gradientNorm%-14.12e")
    else
      logger.info(
        f"Iteration #: $iteration%10d | " +
        f"Func. Eval. #: ${objective.numberOfFunctionEvaluations}%10d | " +
        f"Point Change: $pointChange%-14.12e")
  }

  protected def logTerminationMessage() {
    if (pointConverged)
      logger.info(f"The L2 norm of the point change, $pointChange%-14.12e, " +
                  f"was below the convergence threshold of $pointChangeTolerance%-14.12e.")
    if (objectiveConverged)
      logger.info(f"The relative change of the objective value, $objectiveChange%-14.12e, " +
                  f"was below the convergence threshold of $objectiveChangeTolerance%-14.12e.")
    if (gradientConverged)
      logger.info(f"The gradient norm became $gradientNorm%-14.12e, which is less than " +
                  f"the convergence threshold of $gradientTolerance%-14.12e.")
    if (iteration >= numberOfIterations)
      logger.info(s"Reached the maximum number of iterations, $numberOfIterations.")
    if (objective.numberOfFunctionEvaluations >= numberOfFunctionEvaluations)
      logger.info(s"Reached the maximum number of objective evaluations, $numberOfFunctionEvaluations.")
  }

  protected def performIterationUpdates(): Unit
}

abstract class LineSearchSolver(numberOfIterations: Int = 1000,
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
                                  1.0, ConstantLineSearchInitialization(1.0), 1e-4, 0.9, 10.0
                                ))
  extends IterativeSolver(numberOfIterations,
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
                          logGradientNorm) {
  protected var direction: DenseVector[Double] = null
  protected var stepSize: Double = 0.0
  protected var previousDirection: DenseVector[Double] = null
  protected var previousStepSize: Double = 0.0

  override def minimize(objective: Function, initialPoint: DenseVector[Double]): DenseVector[Double] = {
    gradient = objective.gradient(initialPoint)
    direction = gradient
    previousGradient = gradient
    previousDirection = direction
    super.minimize(objective, initialPoint)
  }

  override protected def performIterationUpdates() = {
    previousDirection = direction
    updateDirection()
    previousStepSize = stepSize
    updateStepSize()
    previousPoint = point
    updatePoint()
    previousGradient = gradient
    gradient = objective.gradient(point)
    if (checkForObjectiveConvergence || logObjectiveValue) {
      previousValue = value
      value = objective.value(point)
    }
  }

  protected def updateStepSize() = {
    stepSize = lineSearch.stepSize(objective, iteration, point, direction,
                                   previousPoint, previousDirection, previousStepSize)
  }

  /**
    *
    *
    * Note: Care must be taken when implementing this method because the previousDirection and the previousPoint
    * variables are simply updated to point to currentDirection and currentPoint respectively, at the beginning of each
    * iteration. That means that when the new values are computed, new objects have to be instantiated for holding
    * those values.
    */
  protected def updateDirection(): Unit

  /**
    *
    *
    * Note: Care must be taken when implementing this method because the previousDirection and the previousPoint
    * variables are simply updated to point to currentDirection and currentPoint respectively, at the beginning of each
    * iteration. That means that when the new values are computed, new objects have to be instantiated for holding
    * those values.
    */
  protected def updatePoint(): Unit
}

abstract class QuasiNewtonSolver(
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
  protected val m: Int = 1
  protected val s = Array.ofDim[DenseVector[Double]](m)
  protected val y = Array.ofDim[DenseVector[Double]](m)
  protected var H: DenseMatrix[Double] = null
  protected var previousH: DenseMatrix[Double] = null

  override def minimize(objective: Function, initialPoint: DenseVector[Double]): DenseVector[Double] = {
    H = DenseMatrix.eye[Double](initialPoint.size)
    super.minimize(objective, initialPoint)
  }

  override protected def updatePoint() = {
    point = previousPoint + direction * stepSize
  }

  protected def updateStoredVectors() = {
    s(0) = point - previousPoint
    y(0) = gradient - previousGradient
  }
}

case class SymmetricRankOneQuasiNewtonSolver(
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
  symmetricRankOneSkippingParameter: Double = 1e-8
) extends QuasiNewtonSolver(numberOfIterations,
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
  override protected def updateDirection() = {
    updateStoredVectors()

  }
}
