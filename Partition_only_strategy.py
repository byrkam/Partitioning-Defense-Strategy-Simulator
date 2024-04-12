import Partitioner
import Evaluator
import Timers
import Logging


def find_malicious_connections(groundTruth, new_partitions, max_time_elapsed, criterion1,
                               min_connections_saved_per, criterion2, max_number_of_nodes, criterion3):

    Timers.total_elapsed_time = 0
    total_connections_saved = 0
    min_connections_saved = min_connections_saved_per * len(groundTruth)
    partition = Partitioner.doPartitioning([groundTruth], new_partitions, criterion1, criterion2, criterion3, max_time_elapsed,
                                           min_connections_saved, max_number_of_nodes, total_connections_saved)
    config = partition[0]
    epoch_time_elapsed = partition[1]
    total_number_of_partitions = len(config)
    number_of_epochs = 1
    epoch_analytics = []

    while True:
        nodesHealthStatus = Evaluator.evaluateConfigurationHealthStatus(config, criterion1, criterion2,
                                                                        max_time_elapsed, min_connections_saved, total_connections_saved)
        if nodesHealthStatus == 0:
            break
        epoch_time_elapsed += nodesHealthStatus[1]

        evaluation = Evaluator.removeHealthyConnections(config, nodesHealthStatus[0])
        config = evaluation[0]
        num_of_nodes_after_eval = len(config)
        total_connections_saved += evaluation[1]
        epoch_analytics.append(Logging.epoch_analytics(groundTruth, config, epoch_time_elapsed))

        all_nodes_are_size_of_one = all(len(node) == 1 for node in config)

        if all_nodes_are_size_of_one:
            break

        if not config or len(config) == 1:
            break

        # new epoch starts
        number_of_epochs += 1
        partition = Partitioner.doPartitioning(config, new_partitions, criterion1, criterion2, criterion3, max_time_elapsed,
                                               min_connections_saved, max_number_of_nodes, total_connections_saved)

        config = partition[0]
        epoch_time_elapsed = partition[1]
        total_number_of_partitions += len(config) - num_of_nodes_after_eval

        if partition[1] == 0:
            break

    if not config:
        print("False alarm. No malicious connections")
    else:
        return Timers.total_elapsed_time, total_number_of_partitions, epoch_analytics, total_connections_saved

