package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.sql.SparkSession
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._
import org.apache.spark.sql.Dataset

case class TrxEdge(
  cust_id_1: Int,
  cust_id_2: Int,
  amount: Float)

/**
 * Contains logic for computing the trx_graph_edge_list featuregroup
 */
object TrxGraphEdgeList {

  case class RawTrx(
    cust_id_in: Int,
    cust_id_out: Int,
    trx_amount: Double,
    trx_bankid: Int,
    trx_clearinnum: Int,
    trx_country: String,
    trx_date: String,
    trx_type: String,
    trx_id: Int)

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
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int, keepInMemory: Boolean = false): Dataset[TrxEdge] = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input).repartition(partitions)
    log.info("Read raw dataframe")
    import spark.implicits._
    val rawDs = rawDf.as[RawTrx]
    log.info("Parsed dataframe to dataset")
    val parsedDs = rawDs.map((trx: RawTrx) => {
      val cust_id_1 = trx.cust_id_in
      val cust_id_2 = trx.cust_id_out
      val amount = trx.trx_amount
      new TrxEdge(cust_id_1, cust_id_2, amount.toFloat)
    })
    if (keepInMemory) {
      return parsedDs
    }
    val featurestore = Hops.getProjectFeaturestore
    val descriptiveStats = true
    val featureCorr = false
    val featureHistograms = true
    val clusterAnalysis = false
    val statColumns = List[String]().asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "The edge list of the transactions graph"
    val primaryKey = "cust_id_1"
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
    return null
  }
}
