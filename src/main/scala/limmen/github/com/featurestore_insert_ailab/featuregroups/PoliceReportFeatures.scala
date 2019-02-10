package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.sql.SparkSession
import java.sql.Timestamp
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._

/**
 * Contains logic for computing the PoliceReportFeatures featuregroup
 */
object PoliceReportFeatures {

  case class RawPoliceReport(
    cust_id: Int,
    description: String,
    report_date: String,
    report_id: Int)

  case class PoliceReportFeature(
    cust_id: Int,
    report_date: Timestamp,
    report_id: Int)

  val formatter = new java.text.SimpleDateFormat("yyyy-MM-dd")

  /**
   * Computes the featuregroup
   *
   * @param input path to the input dataset to read (csv)
   * @param output name of the output featuregroup table
   * @param version version of the featuregroup
   * @param partitions number of spark partitions to parallelize the compute on
   * @param logger spark logger
   * @param create boolean flag whether to create new featuregroup or insert into an existing ones
   * @param jobId the id of the hopsworks job
   */
  def computeFeatures(spark: SparkSession, input: String, featuregroupName: String,
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int): Unit = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input).repartition(partitions)
    log.info("Read raw dataframe")
    import spark.implicits._
    val rawDs = rawDf.as[RawPoliceReport]
    log.info("Parsed dataframe to dataset")
    val parsedDs = rawDs.map((report: RawPoliceReport) => {
      val reportDate = new Timestamp(formatter.parse(report.report_date).getTime)
      val custId = report.cust_id
      val reportId = report.report_id
      PoliceReportFeature(custId, reportDate, reportId)
    })
    val featurestore = Hops.getProjectFeaturestore
    val descriptiveStats = true
    val featureCorr = true
    val featureHistograms = true
    val clusterAnalysis = true
    val statColumns = List[String]().asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "Features on customers reported to the police"
    val primaryKey = "cust_id"
    val dependencies = List[String](input).asJava
    if (create) {
      log.info(s"Creating featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.createFeaturegroup(spark, parsedDs.toDF, featuregroupName, featurestore, version, description,
        jobId, dependencies, primaryKey, descriptiveStats, featureCorr, featureHistograms, clusterAnalysis,
        statColumns, numBins, corrMethod, numClusters)
      log.info(s"Creation of featuregroup $featuregroupName complete")
    } else {
      log.info(s"Inserting into featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.insertIntoFeaturegroup(spark, parsedDs.toDF, featuregroupName, featurestore, version, "overwrite",
        descriptiveStats, featureCorr, featureHistograms, clusterAnalysis, statColumns, numBins, corrMethod,
        numClusters)
      log.info(s"Insertion into featuregroup $featuregroupName complete")
    }
  }
}
