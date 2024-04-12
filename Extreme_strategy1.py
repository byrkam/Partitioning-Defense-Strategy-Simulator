import Partitioner
import Evaluator
import Timers


def run_partitioning_and_evaluation(groundTruth, max_time_elapsed=None, criterion1=None,
                                    min_connections_saved=None, criterion2=None):
    total_number_of_partitions = 0

    Timers.total_elapsed_time = 0

    total_connections_saved = 0

    partition = Partitioner.doPartitioning([groundTruth], len(groundTruth), criterion1, criterion2, max_time_elapsed,
                                           min_connections_saved, total_connections_saved, 0)
    config = partition[0]
    total_number_of_partitions += len(config)
    nodesHealthStatus = Evaluator.evaluateConfigurationHealthStatus(config, criterion1, criterion2,
                                                                    max_time_elapsed, min_connections_saved,
                                                                    total_connections_saved, 0)
    config = Evaluator.removeHealthyConnections(config, nodesHealthStatus)[0]

    return Timers.total_elapsed_time, total_number_of_partitions
