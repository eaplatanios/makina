package makina.scala.optimization

import breeze.linalg.DenseVector
import makina.scala.optimization.LineSearch._

/** Contains methods related to line search algorithms.
  *
  * @author Emmanouil Antonios Platanios
  */
object LineSearch {
  /** Checks whether the Armijo condition (also known as the sufficient decrease condition) is satisfied for a given
    * objective function, point, direction and step size. The Armijo condition makes sure that the reduction in the
    * objective function value is proportional to both the step size and the directional derivative. A typical value
    * for the proportionality constant, {{c}}, is 1e-4.
    *
    * @param  objective         Objective function.
    * @param  point             Point at which to check whether the Armijo condition is satisfied.
    * @param  direction         Direction for which to check whether the Armijo condition is satisfied.
    * @param  stepSize          Step size for which to check whether the Armijo condition is satisfied.
    * @param  c                 Proportionality constant used for the Armijo condition (value in (0,1)).
    * @return Boolean value indicating whether the Armijo condition is satisfied.
    */
  def armijoCondition(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      stepSize: Double,
                      c: Double): Boolean =
    armijoCondition(objective, point, direction, stepSize, c, objective.value(point), objective.gradient(point))

  /** Checks whether the Armijo condition (also known as the sufficient decrease condition) is satisfied for a given
    * objective function, point, direction and step size. The Armijo condition makes sure that the reduction in the
    * objective function value is proportional to both the step size and the directional derivative. A typical value
    * for the proportionality constant, {{c}}, is 1e-4.
    *
    * @param  objective         Objective function.
    * @param  point             Point at which to check whether the Armijo condition is satisfied.
    * @param  direction         Direction for which to check whether the Armijo condition is satisfied.
    * @param  stepSize          Step size for which to check whether the Armijo condition is satisfied.
    * @param  c                 Proportionality constant used for the Armijo condition. Its value should be in (0,1).
    * @param  objectiveValue    Value of the objective function at the current point.
    * @param  objectiveGradient Gradient of the objective function evaluated at the current point.
    * @return Boolean value indicating whether the Armijo condition is satisfied.
    */
  def armijoCondition(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      stepSize: Double,
                      c: Double,
                      objectiveValue: Double,
                      objectiveGradient: DenseVector[Double]): Boolean =
    objective.value(direction * stepSize + point) <= objectiveValue + objectiveGradient.t * direction * c * stepSize

  /** Checks whether the Wolfe conditions are satisfied for a given objective function, point, direction and step size.
    * The Wolfe conditions consist of the Armijo condition (also known as the sufficient decrease condition) and the
    * curvature condition. The Armijo condition makes sure that the reduction in the objective function value is
    * proportional to both the step size and the directional derivative. This condition is satisfied for all
    * sufficiently small values of the step size and so, in order to ensure that the optimization algorithm makes
    * sufficient progress, we also check for the curvature condition. Typical values for the proportionality constants
    * are: for {{c1}}, 1e-4, and for {{c2}}, 0.9 when the search direction is chosen by a Newton or quasi-Newton method
    * and 0.1 when the search direction is obtained from a nonlinear conjugate gradient method.
    *
    * @param  objective         Objective function.
    * @param  point             Point at which to check whether the Wolfe conditions are satisfied.
    * @param  direction         Direction for which to check whether the Wolfe conditions are satisfied.
    * @param  stepSize          Step size for which to check whether the Wolfe conditions are satisfied.
    * @param  c1                Proportionality constant used for the first of the two Wolfe conditions (that is, the
    *                           Armijo condition). Its value should be in (0,1).
    * @param  c2                Proportionality constant used for the second of the two Wolfe conditions (that is, the
    *                           curvature condition). Its value should be in ({{c1}},1).
    * @param  isStrong          Boolean value indicating whether or not to check for the strong Wolfe conditions. The
    *                           only difference is actually on the curvature condition and, when we use the strong Wolfe
    *                           conditions instead of the simple Wolfe conditions, we effectively exclude points from
    *                           the search that are far from the exact line search solution.
    * @return Boolean value indicating whether the Wolfe conditions are satisfied.
    */
  def wolfeConditions(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      stepSize: Double,
                      c1: Double,
                      c2: Double,
                      isStrong: Boolean): Boolean =
    wolfeConditions(objective, point, direction, stepSize, c1, c2, isStrong,
                    objective.value(point), objective.gradient(point))

  /** Checks whether the Wolfe conditions are satisfied for a given objective function, point, direction and step size.
    * The Wolfe conditions consist of the Armijo condition (also known as the sufficient decrease condition) and the
    * curvature condition. The Armijo condition makes sure that the reduction in the objective function value is
    * proportional to both the step size and the directional derivative. This condition is satisfied for all
    * sufficiently small values of the step size and so, in order to ensure that the optimization algorithm makes
    * sufficient progress, we also check for the curvature condition. Typical values for the proportionality constants
    * are: for {{c1}}, 1e-4, and for {{c2}}, 0.9 when the search direction is chosen by a Newton or quasi-Newton method
    * and 0.1 when the search direction is obtained from a nonlinear conjugate gradient method.
    *
    * @param  objective         Objective function.
    * @param  point             Point at which to check whether the Wolfe conditions are satisfied.
    * @param  direction         Direction for which to check whether the Wolfe conditions are satisfied.
    * @param  stepSize          Step size for which to check whether the Wolfe conditions are satisfied.
    * @param  c1                Proportionality constant used for the first of the two Wolfe conditions (that is, the
    *                           Armijo condition). Its value should be in (0,1).
    * @param  c2                Proportionality constant used for the second of the two Wolfe conditions (that is, the
    *                           curvature condition). Its value should be in ({{c1}},1).
    * @param  isStrong          Boolean value indicating whether or not to check for the strong Wolfe conditions. The
    *                           only difference is actually on the curvature condition and, when we use the strong Wolfe
    *                           conditions instead of the simple Wolfe conditions, we effectively exclude points from
    *                           the search that are far from the exact line search solution.
    * @param  objectiveValue    Value of the objective function at the current point.
    * @param  objectiveGradient Gradient of the objective function evaluated at the current point.
    * @return Boolean value indicating whether the Wolfe conditions are satisfied.
    */
  def wolfeConditions(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      stepSize: Double,
                      c1: Double,
                      c2: Double,
                      isStrong: Boolean,
                      objectiveValue: Double,
                      objectiveGradient: DenseVector[Double]): Boolean = {
    // Check for the Armijo condition
    val isArmijoConditionSatisfied: Boolean = armijoCondition(objective, point, direction, stepSize, c1,
                                                              objectiveValue, objectiveGradient)
    // Check for the curvature condition
    val leftTerm = objective.gradient(point + direction * stepSize).t * direction
    val rightTerm = objectiveGradient.t * direction
    (isStrong && isArmijoConditionSatisfied && Math.abs(leftTerm) <= c2 * Math.abs(rightTerm)) ||
    (!isStrong && isArmijoConditionSatisfied && leftTerm >= c2 * rightTerm)
  }

  /** Checks whether the Goldstein conditions are satisfied for a given objective function, point, direction and step
    * size. The Goldstein conditions are similar to the Wolfe conditions and they can also be stated as a pair of
    * inequalities: one inequality corresponding to the Armijo condition (also known as the sufficient decrease
    * condition) and another inequality used to bound the step size from below. However, the second inequality in the
    * case of the Goldstein conditions might exclude all points from the search that are solutions to the exact line
    * search problem. The Goldstein conditions are often used in Newton-type methods but are not well suited for
    * quasi-Newton methods that maintain a positive definite Hessian approximation.
    *
    * @param  objective         Objective function.
    * @param  point             Point at which to check whether the Goldstein conditions are satisfied.
    * @param  direction         Direction for which to check whether the Goldstein conditions are satisfied.
    * @param  stepSize          Step size for which to check whether the Goldstein conditions are satisfied.
    * @param  c                 Proportionality constant used for the Goldstein conditions. Its value should be in
    *                           (0,0.5).
    * @return Boolean value indicating whether the Goldstein conditions are satisfied.
    */
  def goldsteinConditions(objective: Function,
                          point: DenseVector[Double],
                          direction: DenseVector[Double],
                          stepSize: Double,
                          c: Double): Boolean =
    goldsteinConditions(objective, point, direction, stepSize, c, objective.value(point), objective.gradient(point))

  /** Checks whether the Goldstein conditions are satisfied for a given objective function, point, direction and step
    * size. The Goldstein conditions are similar to the Wolfe conditions and they can also be stated as a pair of
    * inequalities: one inequality corresponding to the Armijo condition (also known as the sufficient decrease
    * condition) and another inequality used to bound the step size from below. However, the second inequality in the
    * case of the Goldstein conditions might exclude all points from the search that are solutions to the exact line
    * search problem. The Goldstein conditions are often used in Newton-type methods but are not well suited for
    * quasi-Newton methods that maintain a positive definite Hessian approximation.
    *
    * @param  objective         Objective function.
    * @param  point             Point at which to check whether the Goldstein conditions are satisfied.
    * @param  direction         Direction for which to check whether the Goldstein conditions are satisfied.
    * @param  stepSize          Step size for which to check whether the Goldstein conditions are satisfied.
    * @param  c                 Proportionality constant used for the Goldstein conditions. Its value should be in
    *                           (0,0.5).
    * @param  objectiveValue    Value of the objective function at the current point.
    * @param  objectiveGradient Gradient of the objective function evaluated at the current point.
    * @return Boolean value indicating whether the Goldstein conditions are satisfied.
    */
  def goldsteinConditions(objective: Function,
                          point: DenseVector[Double],
                          direction: DenseVector[Double],
                          stepSize: Double,
                          c: Double,
                          objectiveValue: Double,
                          objectiveGradient: DenseVector[Double]): Boolean = {
    val newObjectiveValue = objective.value(point + direction * stepSize)
    val scaledSearchDirection = objectiveGradient.t * direction * stepSize
    objectiveValue + (1 - c) * scaledSearchDirection <= newObjectiveValue &&
    objectiveValue + c * scaledSearchDirection >= newObjectiveValue
  }
}

/** Trait for line search algorithms.
  *
  * @author Emmanouil Antonios Platanios
  */
trait LineSearch {
  /** Computes the step size value.
    *
    * @param  objective         Objective function.
    * @param  iteration         Current iteration of the solver.
    * @param  point             Point at which we perform the line search.
    * @param  direction         Direction for which we perform the line search.
    * @param  previousPoint     Previous point selected by the optimization algorithm (used by some step size
    *                           initialization methods for iterative line search algorithms).
    * @param  previousDirection Previous direction selected by the optimization algorithm (used by some step size
    *                           initialization methods for iterative line search algorithms).
    * @param  previousStepSize  Previous step size used by the optimization algorithm (used by some step size
    *                           initialization methods for iterative line search algorithms).
    * @return A step size value that satisfies certain criteria which depend on the algorithm choice.
    */
  def stepSize(objective: Function,
               iteration: Int,
               point: DenseVector[Double],
               direction: DenseVector[Double],
               previousPoint: DenseVector[Double],
               previousDirection: DenseVector[Double],
               previousStepSize: Double): Double
}

/** Class for using a fixed step size with line search optimization algorithms.
  *
  * @param  stepSize  Fixed step size value.
  * @author Emmanouil Antonios Platanios
  */
case class NoLineSearch(stepSize: Double = 1.0) extends LineSearch {
  override def stepSize(objective: Function,
                        iteration: Int,
                        point: DenseVector[Double],
                        direction: DenseVector[Double],
                        previousPoint: DenseVector[Double],
                        previousDirection: DenseVector[Double],
                        previousStepSize: Double): Double = stepSize
}

/** Class for using a decaying step size with line search optimization algorithms. The step size is computed as
  * (tau + i + 1) &#94; (-kappa), where i is the current iteration number in the optimization algorithm.
  *
  * @param  tau   Parameter >= 0.
  * @param  kappa Parameter in (0.5,1].
  * @author Emmanouil Antonios Platanios
  */
case class DecayingLineSearch(tau: Double, kappa: Double) extends LineSearch {
  if (tau < 0.0)
    throw new IllegalArgumentException("The value of the tau parameter must be >= 0.")
  if (kappa <= 0.5 || kappa > 1.0)
    throw new IllegalArgumentException("The value of the kappa parameter must be in (0.5,1].")

  override def stepSize(objective: Function,
                        iteration: Int,
                        point: DenseVector[Double],
                        direction: DenseVector[Double],
                        previousPoint: DenseVector[Double],
                        previousDirection: DenseVector[Double],
                        previousStepSize: Double): Double = Math.pow(tau + iteration + 1, -kappa)
}

/** Class for an exact line search algorithm. It currently requires the objective function to be a quadratic function.
  * This line search algorithm currently only supports quadratic objective functions.
  *
  * @author Emmanouil Antonios Platanios
  */
case class ExactLineSearch() extends LineSearch {
  override def stepSize(objective: Function,
                        iteration: Int,
                        point: DenseVector[Double],
                        direction: DenseVector[Double],
                        previousPoint: DenseVector[Double],
                        previousDirection: DenseVector[Double],
                        previousStepSize: Double): Double = {
    -(objective.gradient(point).t * direction) / (direction.t * objective.asInstanceOf[QuadraticFunction].A * direction)
  }
}

/** Abstract class for all iterative line search algorithms.
  *
  * @param  initialStepSize Initial step size in the iterative line search process.
  * @param  initialization  Step size initialization method.
  * @author Emmanouil Antonios Platanios
  */
abstract class IterativeLineSearch(initialStepSize: Double = 1.0,
                                   initialization: LineSearchInitialization = ConstantLineSearchInitialization())
  extends LineSearch {
  if (initialStepSize <= 0.0)
    throw new IllegalArgumentException("The initial step size value must be a positive number.")
  protected var objective: Function = null

  override def stepSize(objective: Function,
                        iteration: Int,
                        point: DenseVector[Double],
                        direction: DenseVector[Double],
                        previousPoint: DenseVector[Double],
                        previousDirection: DenseVector[Double],
                        previousStepSize: Double): Double = {
    this.objective = objective
    lineSearch(
      point,
      direction,
      if (iteration > 0)
        initialization.initialStepSize(objective, point, direction, previousPoint, previousDirection, previousStepSize)
      else
        1.0
    )
  }

  /** Performs the actual line search.
    *
    * @param  point           Point at which we perform the line search.
    * @param  direction       Direction for which we perform the line search.
    * @param  initialStepSize Initial step size for line search.
    * @return A step size value that satisfies certain criteria that depend on the algorithm choice.
    */
  protected def lineSearch(point: DenseVector[Double], direction: DenseVector[Double], initialStepSize: Double): Double
}

/** Class for a simple backtracking line search algorithm. This algorithm starts from some initial step size value and
  * keeps reducing it, by multiplying it with a contraption factor, until it satisfies the Armijo condition (also known
  * as the sufficient decrease condition).
  *
  * @param  initialStepSize   Initial step size in the iterative line search process.
  * @param  initialization    Step size initialization method.
  * @param  contraptionFactor Contraption factor.
  * @param  c                 Proportionality constant for the Armijo condition.
  * @param  tolerance         Minimum allowed value for the step size.
  * @author Emmanouil Antonios Platanios
  */
case class BacktrackingLineSearch(initialStepSize: Double = 1.0,
                                  initialization: LineSearchInitialization = ConstantLineSearchInitialization(),
                                  contraptionFactor: Double,
                                  c: Double,
                                  tolerance: Double = 1e-10)
  extends IterativeLineSearch(initialStepSize, initialization) {
  if (contraptionFactor <= 0.0 || contraptionFactor >= 1.0)
    throw new IllegalArgumentException("The value of the contraction factor must be in (0,1).")
  if (c <= 0.0 || c >= 1.0)
    throw new IllegalArgumentException("The value of the c parameter must be in (0,1).")

  override protected def lineSearch(point: DenseVector[Double],
                                    direction: DenseVector[Double],
                                    initialStepSize: Double): Double = {
    val objectiveValue: Double = objective.value(point)
    val objectiveGradient: DenseVector[Double] = objective.gradient(point)
    var stepSize: Double = initialStepSize
    while (!armijoCondition(objective, point, direction, stepSize, c, objectiveValue, objectiveGradient) &&
           stepSize >= tolerance)
      stepSize *= contraptionFactor
    stepSize
  }
}

/** Class for an interpolation based line search algorithm that returns a step size which satisfies the Armijo condition
  * (also known as the sufficient decrease condition). This is an implementation of an algorithm described in pages
  * 57-58 of the book "Numerical Optimization", by Jorge Nocedal and Stephen Wright.
  *
  * @param  c         Proportionality constant used for the Armijo condition (value in (0,1)).
  * @author Emmanouil Antonios Platanios
  */
case class ArmijoInterpolationLineSearch(initialStepSize: Double = 1.0,
                                         initialization: LineSearchInitialization = ConstantLineSearchInitialization(),
                                         c: Double)
  extends IterativeLineSearch(initialStepSize, initialization) {
  if (c <= 0.0 || c >= 1.0)
    throw new IllegalArgumentException("The value of the c parameter must be in (0,1).")
  /** Threshold for the minimum allowed step size change during an interpolation step. */
  private val minimumStepSizeChange: Double = 1e-3
  /** Threshold for the minimum allowed step size change ratio during an interpolation step. */
  private val minimumStepSizeRatio: Double = 1e-1
  /** Temporary variable used to store the old value of the step size during the iterative line search procedure. */
  /** Temporary variable used  to store the new value of the step size during the iterative line search procedure. */
  private var aOld: Double = 0.0
  private var aNew: Double = 0.0

  override protected def lineSearch(point: DenseVector[Double],
                                    direction: DenseVector[Double],
                                    initialStepSize: Double): Double = {
    val phi0 = objective.value(point)
    val objectiveGradient = objective.gradient(point)
    val phiPrime0 = objectiveGradient.t * direction
    aNew = initialStepSize
    var isFirstIteration = true
    while (!armijoCondition(objective, point, direction, aNew, c, phi0, objectiveGradient)) {
        if (isFirstIteration) {
          interpolateQuadratic(point, direction, phi0, phiPrime0)
          isFirstIteration = false
        } else {
          interpolateQubic(point, direction, phi0, phiPrime0)
        }
    }
    aNew
  }

  /** Performs a quadratic interpolation using the available information in order to obtain an approximation of the
    * &phi; function and returns the step size value that minimizes this approximation. This function is only used for
    * the first iteration of the line search algorithm, when we do not yet have enough information available to perform
    * a cubic interpolation.
    *
    * @param  point     Point at which to perform the line search.
    * @param  direction Direction for which to perform the line search.
    * @param  phi0      Value of &phi;(0) (i.e., value of the objective function at the point where we perform the line
    *                   search).
    * @param  phiPrime0 Value of &phi;'(0) (i.e., value of the objective function gradient at the point where we perform
    *                   the line search).
    */
  private def interpolateQuadratic(point: DenseVector[Double],
                                   direction: DenseVector[Double],
                                   phi0: Double,
                                   phiPrime0: Double) = {
    aOld = aNew
    val phiA0 = objective.value(point + direction * aOld)
    aNew = -phiPrime0 * Math.pow(aOld, 2) / (2 * (phiA0 - phi0 - aOld * phiPrime0))
    // Ensure that we make reasonable progress and that the final step size is not too small
    if (Math.abs(aNew - aOld) <= minimumStepSizeChange || aNew / aOld <= minimumStepSizeRatio)
      aNew = aOld / 2
  }

  /** Performs a cubic interpolation using the available information in order to obtain an approximation of the &phi;
    * function and returns the step size value that minimizes this approximation.
    *
    * @param  point     Point at which to perform the line search.
    * @param  direction Direction for which to perform the line search.
    * @param  phi0      Value of &phi;(0) (i.e., value of the objective function at the point where we perform the line
    *                   search).
    * @param  phiPrime0 Value of &phi;'(0) (i.e., value of the objective function gradient at the point where we perform
    *                   the line search).
    */
  private def interpolateQubic(point: DenseVector[Double],
                               direction: DenseVector[Double],
                               phi0: Double,
                               phiPrime0: Double) = {
    val a0Square = Math.pow(aOld, 2)
    val a1Square = Math.pow(aNew, 2)
    val a0Cube = Math.pow(aOld, 3)
    val a1Cube = Math.pow(aNew, 3)
    val phiA0 = objective.value(point +  direction * aOld)
    val phiA1 = objective.value(point +  direction * aNew)
    val denominator = a0Square * a1Square * (aNew - aOld)
    val a = (a0Square * (phiA1 - phi0 - aNew * phiPrime0) - a1Square * (phiA0 - phi0 - aOld * phiPrime0)) / denominator
    val b = (-a0Cube * (phiA1 - phi0 - aNew * phiPrime0) + a1Cube * (phiA0 - phi0 - aOld * phiPrime0)) / denominator
    aOld = aNew
    aNew = -(b - Math.sqrt(Math.pow(b, 2) - 3 * a * phiPrime0)) / (3 * a)
    // Ensure that we make reasonable progress and that the final step size is not too small
    if (Math.abs(aNew - aOld) <= minimumStepSizeChange || aNew / aOld <= minimumStepSizeRatio)
      aNew = aOld / 2
  }
}

/** Class for an interpolation based line search algorithm that returns a step size which satisfies the strong Wolfe
  * conditions. This is an implementation of algorithms 3.5 and 3.6, described in pages 60-62 of the book "Numerical
  * Optimization", by Jorge Nocedal and Stephen Wright.
  *
  * @param  initialStepSize Initial step size in the iterative line search process.
  * @param  initialization  Step size initialization method.
  * @param  c1              Proportionality constant used for the first of the two Wolfe conditions (that is, the Armijo
  *                         condition). Its value should be in (0,1).
  * @param  c2              Proportionality constant used for the second of the two Wolfe conditions (that is, the
  *                         curvature condition). Its value should be in ({{c1}},1).
  * @param  aMax            Maximum allowed value for the step size.
  * @author Emmanouil Antonios Platanios
  */
case class StrongWolfeInterpolationLineSearch(initialStepSize: Double = 1.0,
                                              initialization: LineSearchInitialization = ConstantLineSearchInitialization(),
                                              c1: Double,
                                              c2: Double,
                                              aMax: Double)
  extends IterativeLineSearch(initialStepSize, initialization) {
  if (c1 <= 0.0 || c1 >= 1.0)
    throw new IllegalArgumentException("The value of the c1 parameter must be in (0,1).")
  if (c2 <= c1 || c2 >= 1.0)
    throw new IllegalArgumentException("The value of the c2 parameter must be in (c1,1).")
  /** Maximum number of allowed line search iterations with no improvement in the objective function value. */
  private val maximumIterationsWithNoObjectiveImprovement: Int = 10
  /** Threshold for the minimum allowed distance between a new step size value, computed using a cubic interpolation
    * approach, and the allowed step size values interval endpoints. */
  private val minimumDistanceFromIntervalEndpoints: Double = 1e-3

  override protected def lineSearch(point: DenseVector[Double],
                                    direction: DenseVector[Double],
                                    initialStepSize: Double): Double = {
    val phi0 = objective.value(point)
    val phiPrime0 = objective.gradient(point).t * direction
    var a0 = 0.0
    var a1 = initialStepSize
    if (a1 <= 0 || a1 >= aMax)
      a1 = aMax / 2
    var isFirstIteration = true
    var isLastIteration = false
    var stepSize: Double = 0.0
    while (!isLastIteration) {
      val a1Point = point + direction * a1
      val phiA1 = objective.value(a1Point)
      val phiA0 = objective.value(point + direction * a0)
      isLastIteration = phiA1 > phi0 + c1 * a1 * phiPrime0 || (phiA1 >= phiA0 && !isFirstIteration)
      if (isLastIteration) {
        stepSize = zoom(point, direction, a0, a1)
      } else {
        val phiPrimeA1 = objective.gradient(a1Point).t * direction
        isLastIteration = Math.abs(phiPrimeA1) <= -c2 * phiPrime0
        if (isLastIteration) {
          stepSize = a1
        } else {
          isLastIteration = phiPrimeA1 >= 0
          if (isLastIteration) {
            stepSize = zoom(point, direction, a1, a0)
          } else {
            a0 = a1
            a1 = 2 * a1
            isLastIteration = a1 > aMax
            if (isLastIteration)
              stepSize = aMax
            isFirstIteration = false
          }
        }
      }
    }
    stepSize
  }

  /** "Zooms in" in the interval {{[aLow, aHigh]}} and searches for a step size within that interval that satisfies the
    * strong Wolfe conditions. In each iteration the interval of possible values for the step size is "shrinked" and a
    * new value to test is chosen each time using cubic interpolation. {{aLow}} and {{aHigh}} are such that:
    * <br><br>
    * <ul>
    * <li>The interval bounded by {{aLow}} and {{aHigh}} contains step lengths that satisfy the strong Wolfe
    * conditions.</li>
    * <li>Among all step sizes generated so far and satisfying the Armijo condition (also known as the sufficient
    * decrease condition), {{aLow}} is the one giving the smallest function value.</li>
    * <li>{{aHigh}} is chosen so that &phi;'({{aLow}})({{aHigh}} - {{aLow}}) &lt; 0.</li>
    * </ul>
    * <br>
    * Each iteration of this function generates a new step size, between {{aLow}} and {{aHigh}} and then replaces one of
    * these endpoints by that new step size, in such a way that the above mentioned properties continue to hold. Once a
    * step size value satisfying the strong Wolfe conditions has been found, that value is returned.
    * <br><br>
    * Note that {{aLow}} does not necessarily have to be smaller than {{aHigh}}.
    *
    * @param  point     Point at which to perform the line search.
    * @param  direction Direction for which to perform the line search.
    * @param  aLow      Low endpoint of the interval of possible values for the step size.
    * @param  aHigh     High endpoint of the interval of possible values for the step size.
    * @return A step size value which lies in the interval {{[aLow, aHigh]}} and satisfies the strong Wolfe conditions.
    */
  private def zoom(point: DenseVector[Double], direction: DenseVector[Double], aLow: Double, aHigh: Double): Double = {
    var _aLow = aLow
    var _aHigh = aHigh
    val phi0 = objective.value(point)
    val phiPrime0 = objective.gradient(point).t * direction
    // Declare variables used in the loop that follows
    var aNew: Double = 0.0
    var phiANew: Double = 0.0
    var phiALow: Double = 0.0
    var phiPrimeANew: Double = 0.0
    var aNewPoint: DenseVector[Double] = DenseVector.zeros[Double](0)
    // Declare and initialize variables used to test for convergence of the objective function value
    var minimumObjectiveValue: Double = Double.MaxValue
    var minimumObjectiveValueIteration: Int = -1
    var iteration: Int = 0
    var isLastIteration = false
    while (iteration == 0 || !isLastIteration) {
      aNew = interpolateCubic(point, direction, _aLow, _aHigh)
      aNewPoint = point + direction * aNew
      phiANew = objective.value(aNewPoint)
      phiALow = objective.value(point + direction * _aLow)
      if (phiANew > phi0 + c1 * aNew * phiPrime0 || phiANew >= phiALow) {
        _aHigh = aNew
      } else {
        phiPrimeANew = objective.gradient(aNewPoint).t * direction
        if (Math.abs(phiPrimeANew) <= -c2 * phiPrime0)
          isLastIteration = true
        else if (phiPrimeANew * (_aHigh - _aLow) >= 0)
          _aHigh = _aLow
        _aLow = aNew
      }
      iteration += 1
      if (phiANew < minimumObjectiveValue) {
        minimumObjectiveValue = phiANew
        minimumObjectiveValueIteration = iteration
      } else if (iteration - minimumObjectiveValueIteration > maximumIterationsWithNoObjectiveImprovement) {
        isLastIteration = true
      }
    }
    aNew
  }

  /** Performs a cubic interpolation using the available information in order to obtain an approximation of the &phi;
    * function and returns the step size value, in the interval {{[aLow, aHigh]}}, which minimizes that approximation.
    *
    * @param  point     Point at which to perform the line search.
    * @param  direction Direction for which to perform the line search.
    * @param  aLow      Low endpoint of the interval of possible values for the step size.
    * @param  aHigh     High endpoint of the interval of possible values for the step size.
    * @return A point in the interval {{[aLow, aHigh]}} which minimizes a cubic interpolation approximation of the &phi;
    *         function computed using available information.
    */
  private def interpolateCubic(point: DenseVector[Double],
                               direction: DenseVector[Double],
                               aLow: Double,
                               aHigh: Double): Double = {
    val newPointLow = point + direction * aLow
    val newPointHigh = point + direction * aHigh
    val phiALow = objective.value(newPointLow)
    val phiAHigh = objective.value(newPointHigh)
    val phiPrimeALow = objective.gradient(newPointLow).t * direction
    val phiPrimeAHigh = objective.gradient(newPointHigh).t * direction
    val d1 = phiPrimeALow + phiPrimeAHigh - 3 * (phiALow - phiAHigh) / (aLow - aHigh)
    val d2 = Math.signum(aHigh - aLow) * Math.sqrt(Math.pow(d1, 2) - phiPrimeALow * phiPrimeAHigh)
    var aNew = aHigh - (aHigh - aLow) * (phiPrimeAHigh + d2 - d1) / (phiPrimeAHigh - phiPrimeALow + 2 * d2)
    // Check whether the minimizer is one of the endpoints of the interval or if it is the newly computed value
    if (aLow <= aNew && aNew <= aHigh) {
      val phiANew = objective.value(point + direction * aNew)
      if (phiALow <= phiANew && phiALow <= phiAHigh) aNew = aLow
      else if (phiALow <= phiANew) aNew = aHigh
      else if (phiAHigh <= phiANew) aNew = aHigh
    } else {
      if (phiALow <= phiAHigh) aNew = aLow
      else aNew = aHigh
    }
    // Ensure that the new step size is not too close to the endpoints of the interval
    if (Math.abs(aNew - aLow) <= minimumDistanceFromIntervalEndpoints ||
        Math.abs(aNew - aHigh) <= minimumDistanceFromIntervalEndpoints)
      aNew = (aLow + aHigh) / 2
    aNew
  }
}

/** Abstract class for line search initialization methods, used for computing the initial step size value for line
  * search algorithms. For methods that do not produce well scaled search directions, such as the steepest descent and
  * conjugate gradient methods, it is important to use current information about the problem and the algorithm to make
  * the initial guess. For Newton and quasi-Newton methods, the [[ConstantLineSearchInitialization]] step size
  * initialization method should always be used, with the default fixed step size value of 1.0. This choice ensures that
  * unit step lengths are taken whenever they satisfy the termination conditions and allows the rapid rate of
  * convergence properties of these methods to take effect.
  *
  * @param  initialStepSize Initial step size value chosen by the user.
  * @author Emmanouil Antonios Platanios
  */
abstract class LineSearchInitialization(initialStepSize: Double = 1.0) {
  /** Computes the initial step size.
    *
    * @param   objective          Objective function of the optimization problem.
    * @param   point              Current iterate/point.
    * @param   direction          Current direction selected by the optimization algorithm.
    * @param   previousPoint      Iterate/point of the previous iteration.
    * @param   previousDirection  Direction used by the optimization algorithm in the previous iteration.
    * @param   previousStepSize   Step size used by the optimization algorithm in the previous iterations.
    * @return Initial step size, to be used by line search algorithms.
    */
  def initialStepSize(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      previousPoint: DenseVector[Double],
                      previousDirection: DenseVector[Double],
                      previousStepSize: Double): Double
}

/** Initializes the step size to the provided constant value.
  *
  * @param  initialStepSize Step size value.
  * @author Emmanouil Antonios Platanios
  */
case class ConstantLineSearchInitialization(initialStepSize: Double = 1.0)
  extends LineSearchInitialization(initialStepSize) {
  def initialStepSize(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      previousPoint: DenseVector[Double],
                      previousDirection: DenseVector[Double],
                      previousStepSize: Double): Double = initialStepSize
}

/** Initializes the step size by assuming that the first order change in the objective function at the current
  * iterate/point will be the same as the one obtained in the previous step.
  *
  * @author Emmanouil Antonios Platanios
  */
case class ConserveFirstOrderChangeLineSearchInitialization() extends LineSearchInitialization() {
  def initialStepSize(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      previousPoint: DenseVector[Double],
                      previousDirection: DenseVector[Double],
                      previousStepSize: Double): Double =
    (objective.gradient(previousPoint).t * previousDirection) *
    (previousStepSize / (objective.gradient(point).t * direction))
}

/** Initializes the step size by setting it to equal to the minimizer of a quadratic interpolation using the current
  * information: the objective function value at the current iterate/point, the objective function value at the previous
  * iterate/point, the objective function gradient at the previous iterate/point and the previous direction used by the
  * algorithm.
  *
  * @author Emmanouil Antonios Platanios
  */
case class QuadraticInterpolationLineSearchInitializationMethod() extends LineSearchInitialization() {
  def initialStepSize(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      previousPoint: DenseVector[Double],
                      previousDirection: DenseVector[Double],
                      previousStepSize: Double): Double =
    2 * (objective.value(point) - objective.value(previousPoint)) /
    (objective.gradient(previousPoint).t * previousDirection)
}

/** Initializes the step size by setting it to equal to the minimum between 1 and 1.01 times the minimizer of a
  * quadratic interpolation using the current information: the objective function value at the current iterate/point,
  * the objective function value at the previous iterate/point, the objective function gradient at the previous
  * iterate/point and the previous direction used by the algorithm.
  *
  * @author Emmanouil Antonios Platanios
  */
case class ModifiedQuadraticInterpolationLineSearchInitializationMethod() extends LineSearchInitialization() {
  def initialStepSize(objective: Function,
                      point: DenseVector[Double],
                      direction: DenseVector[Double],
                      previousPoint: DenseVector[Double],
                      previousDirection: DenseVector[Double],
                      previousStepSize: Double): Double =
    Math.min(1.0, 2.02 * (objective.value(point) - objective.value(previousPoint)) /
                  (objective.gradient(previousPoint).t * previousDirection))
}
