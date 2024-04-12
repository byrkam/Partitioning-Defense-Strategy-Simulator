import Timers


def evaluateNodeHealthStatus(node):
    # Check if any connection inside a node is malicious
    for index, connection_status in enumerate(node):
        if connection_status == 1:
            return 1
    return 0


def evaluateConfigurationHealthStatus(configuration, criterion1, criterion2, max_time_elapsed, min_connections_saved,
                                      total_connections_saved, strategy=1):

    if criterion1 == "YES" and Timers.total_elapsed_time > max_time_elapsed:
        return 0
    if criterion2 == "YES" and total_connections_saved > min_connections_saved:
        return 0

    if not configuration:
        return
    health_status = [0] * len(configuration)
    i = 0

    # Check if all nodes in a configuration are healthy
    for node in configuration:
        health_status[i] = evaluateNodeHealthStatus(node=node)
        i = i + 1
    num_of_nodes = 1 if strategy == 0 else len(configuration)
    Timers.increaseTotalElapsedTime(Timers.getRandomEvalTime(1))
    return health_status, Timers.getRandomEvalTime(1)


def removeHealthyConnections(configuration, health_status):
    new_configuration = []
    saved_conns = 0
    if not configuration:
        return
    # Use zip to iterate through both lists simultaneously
    for node, status in zip(configuration, health_status):
        if status == 1:
            new_configuration.append(node)
        elif status == 0:
            saved_conns += len(node)
    return new_configuration, saved_conns

