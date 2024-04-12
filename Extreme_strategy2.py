import Migration
import Evaluator
import Timers


def maxTimeStrategy(groundTruth, max_time_elapsed=None, criterion1=None,
                    min_connections_saved=None, criterion2=None):
    Timers.total_elapsed_time = 0
    index = 0
    total_connections_saved = 0
    while index < len(groundTruth):
        new_node = Migration.migrate(groundTruth, 1, index)
        nodesHealthStatus = Evaluator.evaluateConfigurationHealthStatus([new_node], criterion1, criterion2,
                                                                        max_time_elapsed, min_connections_saved,
                                                                        total_connections_saved, 0)
        new_node = Evaluator.removeHealthyConnections([new_node], nodesHealthStatus)[0]
        index += 1
    return Timers.total_elapsed_time
