package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.sql.SparkSession
import java.sql.Timestamp
import io.hops.util.Hops
import org.apache.spark.sql.Row
import scala.collection.JavaConversions._
import collection.JavaConverters._

/**
 * Contains logic for computing the alert_features featuregroup
 */
object AlertFeatures {

  case class RawAlert(
    alert_date: String,
    alert_desc: String,
    alert_id: Int,
    alert_score: Double,
    alert_type: String,
    rule_name: String,
    trx_id: Int)

  case class AlertFeature(
    alert_date: Timestamp,
    alert_id: Int,
    alert_score: Float,
    alert_type: Long,
    rule_name: Long,
    trx_id: Int)

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
   * @param alertTypeLookup optional pre-computed map with alert types categorical to numeric encoding
   */
  def computeFeatures(spark: SparkSession, input: String, featuregroupName: String, version: Int,
    partitions: Int, log: Logger, create: Boolean, jobId: Int, alertTypeLookup: Map[String, Long] = null,
    ruleNameLookup: Map[String, Long] = null): Unit = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input).repartition(partitions)
    import spark.implicits._
    val rawDs = rawDf.as[RawAlert]
    val featurestore = Hops.getProjectFeaturestore
    var alertTypeLookupMap: Map[String, Long] = null
    if (alertTypeLookup == null) {
      val alertTypeLookupDf = Hops.getFeaturegroup(spark, "alert_type_lookup", featurestore, 1)
      val alertTypeLookupList: Array[Row] = alertTypeLookupDf.collect
      alertTypeLookupMap = alertTypeLookupList.map((row: Row) => {
        row.getAs[String]("alert_type") -> row.getAs[Long]("id")
      }).toMap
    } else {
      alertTypeLookupMap = alertTypeLookup
    }
    var ruleNameLookupMap: Map[String, Long] = null
    if (ruleNameLookup == null) {
      val ruleNameLookupDf = Hops.getFeaturegroup(spark, "rule_name_lookup", featurestore, 1)
      val ruleNameLookupList: Array[Row] = ruleNameLookupDf.collect
      ruleNameLookupMap = ruleNameLookupList.map((row: Row) => {
        row.getAs[String]("rule_name") -> row.getAs[Long]("id")
      }).toMap
    } else {
      ruleNameLookupMap = ruleNameLookup
    }
    val parsedDs = rawDs.map((alert: RawAlert) => {
      val alertDate = new Timestamp(formatter.parse(alert.alert_date).getTime)
      val alertId = alert.alert_id
      val alertScore = alert.alert_score.toFloat
      val alertType = alertTypeLookupMap(alert.alert_type)
      val ruleName = ruleNameLookupMap(alert.rule_name)
      val trxId = alert.trx_id
      AlertFeature(alertDate, alertId, alertScore, alertType, ruleName, trxId)
    })
    val descriptiveStats = true
    val featureCorr = true
    val featureHistograms = true
    val clusterAnalysis = true
    val statColumns = List[String]().asJava
    val dependencies = List[String](input).asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "Features from transaction alerts"
    val primaryKey = "alert_id"
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
