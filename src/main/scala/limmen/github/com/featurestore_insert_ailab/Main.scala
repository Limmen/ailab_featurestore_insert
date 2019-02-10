package limmen.github.com.feature_engineering_spark

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ Row, SparkSession }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.sql.DataFrame;
import org.rogach.scallop.ScallopConf
import limmen.github.com.feature_engineering_spark._
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._

/**
 * Parser of command-line arguments
 */
class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val input = opt[String](required = false, descr = "path to input config (csv)")
  verify()
}

case class Entity(
  name: String,
  inputFile: String,
  description: String,
  clusterAnalysis: Boolean,
  featureHistograms: Boolean,
  descriptiveStats: Boolean,
  featureCorr: Boolean,
  primaryKey: String,
  entityType: String,
  dataFormat: String)

object Main {

  def main(args: Array[String]): Unit = {

    // Setup logging
    val log = LogManager.getRootLogger()
    log.setLevel(Level.INFO)
    log.info(s"Starting Job to Insert AILAB data into Feature Groups/Training Dataset")

    //Parse cmd arguments
    val conf = new Conf(args)

    val configPath = conf.input()

    // Setup Spark
    val sparkConf = sparkClusterSetup()
    val spark = SparkSession.builder().config(sparkConf).enableHiveSupport().getOrCreate();
    val sc = spark.sparkContext
    val clusterStr = sc.getConf.toDebugString
    log.info(s"Cluster settings: \n" + clusterStr)

    import spark.implicits._
    log.info("Reading Config")
    val configDf = readConfigFile(configPath = configPath, spark = spark)
    val configDs = configDf.as[Entity]
    val configList = configDs.collect()
    val featurestore = Hops.getProjectFeaturestore
    log.info("Starting the transfer")
    configList.foreach((entity: Entity) => {
      log.info(s"Creating featuregroup ${entity.name} version in featurestore $featurestore")
      val data = spark.read.parquet(entity.inputFile)
      val statColumns = List[String]().asJava
      val dependencies = List[String](entity.inputFile).asJava
      val numBins = 20
      val corrMethod = "pearson"
      val numClusters = 5
      if (entity.entityType.equalsIgnoreCase("featuregroup")) {
        log.info(s"Creating featuregroup ${entity.name} version in featurestore $featurestore")
        Hops.createFeaturegroup(spark, data, entity.name, featurestore, 1, entity.description,
          null, dependencies, entity.primaryKey, entity.descriptiveStats, entity.featureCorr, entity.featureHistograms, entity.clusterAnalysis,
          statColumns, numBins, corrMethod, numClusters)
      } else {
        log.info(s"Creating training dataset ${entity.name} version in featurestore $featurestore")
        Hops.createTrainingDataset(spark, data, entity.name, featurestore, 1, entity.description,
          null, entity.dataFormat, dependencies, entity.descriptiveStats, entity.featureCorr, entity.featureHistograms, entity.clusterAnalysis,
          statColumns, numBins, corrMethod, numClusters)
      }
    })
    log.info("Shutting down spark job")
    spark.close
  }

  def readConfigFile(configPath: String, spark: SparkSession): DataFrame = {
    spark.read.format("csv").option("header", "true").option("delimiter", ",").load(configPath)
  }

  /**
   * Hard coded settings for local spark training
   *
   * @return spark configurationh
   */
  def localSparkSetup(): SparkConf = {
    new SparkConf().setAppName("feature_engineering_spark").setMaster("local[*]")
  }

  /**
   * Hard coded settings for cluster spark training
   *
   * @return spark configuration
   */
  def sparkClusterSetup(): SparkConf = {
    new SparkConf().setAppName("feature_engineering_spark").set("spark.executor.heartbeatInterval", "20s").set("spark.rpc.message.maxSize", "512").set("spark.kryoserializer.buffer.max", "1024")
  }

}
