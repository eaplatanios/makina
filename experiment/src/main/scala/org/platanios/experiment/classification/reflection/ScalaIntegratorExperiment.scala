package org.platanios.experiment.classification.reflection

import java.text.DecimalFormat

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

class ScalaIntegratorExperiment {

}

object ScalaIntegratorExperiment {
  val DECIMAL_FORMAT = new DecimalFormat("0.0000000000E0")
  val RESULTS_DECIMAL_FORMAT = new DecimalFormat("0.00E0")
  val logger = Logger(LoggerFactory.getLogger("Classification / Reflection / Integrator Experiment"))

  def main(args: Array[String]): Unit = {
    logger.info("Hello, world!")
  }
}
