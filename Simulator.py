import argparse
import Params
import Extreme_strategy1
import Extreme_strategy2
import Partition_only_strategy
from itertools import zip_longest
import Logging
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Run a network defense simulation.')

    parser.add_argument('--strategy', type=str, help='Strategy to execute: 1: Every connections is migrated to a VM, '
                                                     '2: Only 1 VM is spawned, 3: Only partitions strategy, '
                                                     '4: Random shuffle strategy')
    parser.add_argument('--num_tests', type=int, default=Params.NUMBEROFTESTS, help='Number of simulation tests')
    # Add more command line arguments as needed
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    time_elapsed_per_test = []
    nodes_spawned_per_test = []
    total_VMs_spawned = 0
    number_of_connections = Params.NUMBEROFCONNECTIONS
    pollution_factors = Params.POLUTIONFACTOR
    groundTruths = Params.GROUNDTRUTH
    max_time_elapsed = Params.MAXTIMEELAPSEDGLOBAL
    criterion1 = Params.MAXTIMECRITERION
    per_of_conns_to_migrate = Params.PERCENTAGEOFCONNSTOMIGRATE
    new_partitions = Params.NUMBEROFNODES
    max_number_of_nodes = Params.MAXCOSTGLOBAL
    criterion3 = Params.MAXCOSTCRITERION
    min_connections_saved_per = Params.MINCONNECTIONSSAFEGLOBALPC
    criterion2 = Params.CONNECTIONSSAFECRITERION
    type_of_gTruth = Params.GROUNDTRUTHGENTYPE
    num_of_tests = args.num_tests
    choice = int(args.strategy)

    if choice == 1:
        for index, pol_factor in zip(pollution_factors, zip(*groundTruths)):
            # Write a delimiter line
            with open("simulation_res/N_replicas/N_replicas.txt", "a") as file:
                file.write("Distribution: " + type_of_gTruth + "-" * 10 + "Pollution factor: " + str(index) + "-" * 10 +
                           "\n")
            for ground_truth in pol_factor:
                time_elapsed_per_test.clear()
                for _ in range(num_of_tests):
                    total_elapsed_time, total_VMs_spawned = Extreme_strategy1.run_partitioning_and_evaluation(ground_truth)
                    time_elapsed_per_test.append(total_elapsed_time)

                average_elapsed_time = sum(time_elapsed_per_test) / len(time_elapsed_per_test)

                # Save the average times to a text file
                with open("simulation_res/N_replicas/N_replicas.txt", "a") as file:
                    file.write(
                        f"GroundTruth Length: {len(ground_truth)}, Average Elapsed Time = {average_elapsed_time}, "
                        f"Total VMs spawned = {total_VMs_spawned}\n")
        Logging.split_file_by_delimiter('simulation_res/N_replicas/N_replicas.txt', 'simulation_res/N_replicas/N_replicas')
        os.remove('simulation_res/N_replicas/N_replicas.txt')
    elif choice == 2:
        for index, pol_factor in zip(pollution_factors, zip(*groundTruths)):
            # Write a delimiter line
            with open("simulation_res/One_replica/One_replica.txt", "a") as file:
                file.write("Distribution: " + type_of_gTruth + "-" * 10 + "Pollution factor: " + str(index) + "-" * 10 +
                           "\n")
            for ground_truth in pol_factor:
                time_elapsed_per_test.clear()
                for _ in range(num_of_tests):
                    total_elapsed_time = Extreme_strategy2.maxTimeStrategy(ground_truth)
                    time_elapsed_per_test.append(total_elapsed_time)
                average_elapsed_time = sum(time_elapsed_per_test) / len(time_elapsed_per_test)

                # Save the average times to a text file
                with open("simulation_res/One_replica/One_replica.txt", "a") as file:
                    file.write(
                        f"GroundTruth Length: {len(ground_truth)}, Average Elapsed Time = {average_elapsed_time},"
                        f" Total VMs spawned = 1\n")
        Logging.split_file_by_delimiter('simulation_res/One_replica/One_replica.txt', 'simulation_res/One_replica/One_replica')
        os.remove('simulation_res/One_replica/One_replica.txt')
    elif choice == 3:
        total_connections_saved_per_test = []
        connections_saved_per_test = []
        partitions_per_test = []
        average_epochs_per_test = []
        malicious_conns_percentage_per_test = []
        nodes_health_percentage_per_epoch = []
        epoch_time_elapsed_per_test = []
        for pol_per, pol_factor in zip(pollution_factors, zip(*groundTruths)):
            for partition in new_partitions:
                # Write a delimiter line
                with open("simulation_res/partitions_only/partitions_only.txt", "a") as file:
                    file.write("Distribution: " + type_of_gTruth + "-" * 10 + "Pollution factor: " + str(pol_per) +
                               ", Partition step: " + str(partition) + "-" * 10 + "\n")
                for ground_truth in pol_factor:
                    time_elapsed_per_test.clear()
                    total_connections_saved_per_test.clear()
                    nodes_spawned_per_test.clear()
                    connections_saved_per_test.clear()
                    partitions_per_test.clear()
                    average_epochs_per_test.clear()
                    malicious_conns_percentage_per_test.clear()
                    nodes_health_percentage_per_epoch.clear()
                    epoch_time_elapsed_per_test.clear()
                    for _ in range(num_of_tests):
                        metrics = Partition_only_strategy.find_malicious_connections(ground_truth, partition,
                                                                                     max_time_elapsed, criterion1,
                                                                                     min_connections_saved_per,
                                                                                     criterion2,
                                                                                     max_number_of_nodes, criterion3)

                        time_elapsed_per_test.append(metrics[0])
                        nodes_spawned_per_test.append(metrics[1])
                        total_connections_saved_per_test.append(metrics[3])
                        average_epochs = len(metrics[2])
                        average_epochs_per_test.append(average_epochs)
                        epoch_time_elapsed_for_current_test = [epoch[3] if len(epoch) > 0 else None for epoch in
                                                               metrics[2]]
                        epoch_time_elapsed_per_test.append(epoch_time_elapsed_for_current_test)
                        connections_saved_for_current_test = [epoch[0] if len(epoch) > 0 else None for epoch in
                                                              metrics[2]]
                        connections_saved_per_test.append(connections_saved_for_current_test)
                        partitions_per_epoch = [epoch[1] if len(epoch) > 0 else None for epoch in metrics[2]]
                        partitions_per_test.append(partitions_per_epoch)
                        malicious_conns_percentage_per_epoch = [epoch[2] if len(epoch) > 0 else None for epoch in
                                                                metrics[2]]
                        malicious_conns_percentage_per_test.append(malicious_conns_percentage_per_epoch)

                        transposed_data = list(zip_longest(*malicious_conns_percentage_per_test, fillvalue=0))
                        max_sizes = [max(map(len, tpl)) if all(isinstance(lst, list) for lst in tpl) else 1 for tpl
                                     in
                                     transposed_data]
                        adjusted_data = [
                            tuple(
                                (lst + [0] * (max_size - len(lst))) if isinstance(lst, list) else (
                                        [0] * (max_size - 1) + [lst])
                                for lst, max_size in zip(tpl, [max_sizes[i]] * len(tpl))
                            )
                            for i, tpl in enumerate(transposed_data)
                        ]
                        summed_data = [list(map(sum, zip(*tpl))) for tpl in adjusted_data]

                    average_epochs = sum(average_epochs_per_test) / len(average_epochs_per_test)
                    average_time_elapsed_per_epoch = [sum(x) / num_of_tests for x in zip(*epoch_time_elapsed_per_test)]
                    average_partitions_per_epoch = [sum(x) / num_of_tests for x in zip(*partitions_per_test)]
                    average_connections_saved_per_epoch = [sum(x) / num_of_tests for x in
                                                           zip(*connections_saved_per_test)]
                    average_pollution_rate_per_node_per_epoch = [[value / num_of_tests for value in row] for row in
                                                                 summed_data]

                    average_elapsed_time = sum(time_elapsed_per_test) / len(time_elapsed_per_test)
                    average_VMs_spawned = int(sum(nodes_spawned_per_test) / len(nodes_spawned_per_test))

                    average_connections_saved = (sum(total_connections_saved_per_test) / len(
                        total_connections_saved_per_test)) / len(ground_truth)
                    # Save the average times to a text file
                    with open(f"simulation_res/partitions_only/partitions_only.txt", "a") as file:
                        file.write(
                            f"GroundTruth Length: {len(ground_truth)}, Average Elapsed Time = {average_elapsed_time}, "
                            f"Total VMs spawned = {average_VMs_spawned}, "
                            f"Average (%) connections saved = {average_connections_saved}\n")
                    with open("simulation_res/partitions_only/partitions_only_epoch_analytics.txt", "a") as file:
                        file.write(
                            f"{'-' * 10} GroundTruth Length = {len(ground_truth)}, Pollution factor = {pol_per}, "
                            f"Partition step = {partition} {'-' * 10}\n"
                            f"Average epochs: {average_epochs}\n"
                            f"Average time elapsed per epoch: {average_time_elapsed_per_epoch}\n"
                            f"Average partitions per epoch: {average_partitions_per_epoch}\n"
                            f"Average connections saved per epoch: {average_connections_saved_per_epoch}\n")
                        for index, epoch in enumerate(average_pollution_rate_per_node_per_epoch):
                            file.write(f"Average pollution rate per node in epoch {index + 1}: {epoch}\n")
        Logging.split_file_by_delimiter('simulation_res/partitions_only/partitions_only.txt',
                                        'simulation_res/partitions_only/partitions_only')
        os.remove('simulation_res/partitions_only/partitions_only.txt')
