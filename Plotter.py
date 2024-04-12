import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from itertools import accumulate


def plot_average_times(criterion, min_filename, max_filename, only_partition_filename, only_partition_exp_filename,
                       only_partition_rev_exp_filename, only_partition_add_filename, only_partition_sub_filename,
                       distr_based_only_partition_filename):
    # Initialize lists to store data from both files
    min_ground_truth_lengths = []
    min_average_times = []

    max_ground_truth_lengths = []
    max_average_times = []

    partitions_only_exp_ground_truth_lengths = []
    partitions_only_exp_average_times = []

    partitions_only_rev_exp_ground_truth_lengths = []
    partitions_only_rev_exp_average_times = []

    partitions_only_sub_ground_truth_lengths = []
    partitions_only_sub_average_times = []

    partitions_only_add_ground_truth_lengths = []
    partitions_only_add_average_times = []

    distr_based_partitions_only_ground_truth_lengths = []
    distr_based_partitions_only_average_times = []

    partition_only_steps = {}  # Store data for partition only strategy

    min_total_vms_spawned = []  # New list for Total VMs spawned in the minimum file
    max_total_vms_spawned = []  # New list for Total VMs spawned in the maximum file
    partitions_only_exp_total_vms_spawned = []
    partitions_only_rev_exp_total_vms_spawned = []
    partitions_only_sub_total_vms_spawned = []
    partitions_only_add_total_vms_spawned = []
    distr_based_partitions_only_total_vms_spawned = []

    partitions_only_exp_conns_saved = []
    partitions_only_rev_exp_conns_saved = []
    distr_based_partitions_only_conns_saved = []
    partitions_only_sub_conns_saved = []
    partitions_only_add_conns_saved = []

    # Read data from the minimum time file
    with open(min_filename, "r") as min_file:
        lines = min_file.readlines()
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 3:
                length = int(parts[0].split(": ")[1])
                avg_time = float(parts[1].split(" = ")[1]) / 60000
                min_total_vms = int(parts[2].split(" = ")[1])  # New parameter
                min_ground_truth_lengths.append(length)
                min_average_times.append(avg_time)
                min_total_vms_spawned.append(min_total_vms)  # Append to the new list

    with open(max_filename, "r") as max_file:
        lines = max_file.readlines()
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 3:
                length = int(parts[0].split(": ")[1])
                avg_time = float(parts[1].split(" = ")[1]) / 60000
                max_total_vms = int(parts[2].split(" = ")[1])  # New parameter
                max_ground_truth_lengths.append(length)
                max_average_times.append(avg_time)
                max_total_vms_spawned.append(max_total_vms)  # Append to the new list

    with open(only_partition_exp_filename, "r") as exp_file:
        lines = exp_file.readlines()
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 4:
                length = int(parts[0].split(": ")[1])
                avg_time = float(parts[1].split(" = ")[1]) / 60000
                min_total_vms = int(parts[2].split(" = ")[1])  # New parameter
                conns_saved = float(parts[3].split(" = ")[1])
                partitions_only_exp_ground_truth_lengths.append(length)
                partitions_only_exp_average_times.append(avg_time)
                partitions_only_exp_total_vms_spawned.append(min_total_vms)  # Append to the new list
                partitions_only_exp_conns_saved.append(conns_saved)

    with open(only_partition_rev_exp_filename, "r") as rev_exp_file:
        lines = rev_exp_file.readlines()
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 4:
                length = int(parts[0].split(": ")[1])
                avg_time = float(parts[1].split(" = ")[1]) / 60000
                min_total_vms = int(parts[2].split(" = ")[1])  # New parameter
                conns_saved = float(parts[3].split(" = ")[1])
                partitions_only_rev_exp_ground_truth_lengths.append(length)
                partitions_only_rev_exp_average_times.append(avg_time)
                partitions_only_rev_exp_total_vms_spawned.append(min_total_vms)  # Append to the new list
                partitions_only_rev_exp_conns_saved.append(conns_saved)

    with open(distr_based_only_partition_filename, "r") as distr_based_file:
        lines = distr_based_file.readlines()
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 4:
                length = int(parts[0].split(": ")[1])
                avg_time = float(parts[1].split(" = ")[1]) / 60000
                min_total_vms = int(parts[2].split(" = ")[1])  # New parameter
                conns_saved = float(parts[3].split(" = ")[1])
                distr_based_partitions_only_ground_truth_lengths.append(length)
                distr_based_partitions_only_average_times.append(avg_time)
                distr_based_partitions_only_total_vms_spawned.append(min_total_vms)  # Append to the new list
                distr_based_partitions_only_conns_saved.append(conns_saved)

    with open(only_partition_sub_filename, "r") as sub_file:
        lines = sub_file.readlines()
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 4:
                length = int(parts[0].split(": ")[1])
                avg_time = float(parts[1].split(" = ")[1]) / 60000
                min_total_vms = int(parts[2].split(" = ")[1])  # New parameter
                conns_saved = float(parts[3].split(" = ")[1])
                partitions_only_sub_ground_truth_lengths.append(length)
                partitions_only_sub_average_times.append(avg_time)
                partitions_only_sub_total_vms_spawned.append(min_total_vms)  # Append to the new list
                partitions_only_sub_conns_saved.append(conns_saved)

    with open(only_partition_add_filename, "r") as add_file:
        lines = add_file.readlines()
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 4:
                length = int(parts[0].split(": ")[1])
                avg_time = float(parts[1].split(" = ")[1]) / 60000
                min_total_vms = int(parts[2].split(" = ")[1])  # New parameter
                conns_saved = float(parts[3].split(" = ")[1])
                partitions_only_add_ground_truth_lengths.append(length)
                partitions_only_add_average_times.append(avg_time)
                partitions_only_add_total_vms_spawned.append(min_total_vms)  # Append to the new list
                partitions_only_add_conns_saved.append(conns_saved)

    with open(only_partition_filename, "r") as only_part_file:
        lines = only_part_file.readlines()
        current_partitioning_step = None
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 2:
                match = re.search(r'\d+', parts[1])
                current_partitioning_step = int(match.group())
                partition_only_steps[current_partitioning_step] = {"ground_truth_lengths": [],
                                                                   "average_times": [],
                                                                   "total_vms_spawned": [],
                                                                   "conns_saved": []}
            elif current_partitioning_step is not None and line.startswith("GroundTruth Length:"):
                length = int(parts[0].split(": ")[1])
                avg_time = float(parts[1].split(" = ")[1]) / 60000
                only_part_total_vms = int(parts[2].split(" = ")[1])  # New parameter
                conns_saved = float(parts[3].split(" = ")[1])
                partition_only_steps[current_partitioning_step]["ground_truth_lengths"].append(length)
                partition_only_steps[current_partitioning_step]["average_times"].append(avg_time)
                partition_only_steps[current_partitioning_step]["total_vms_spawned"].append(only_part_total_vms)
                partition_only_steps[current_partitioning_step]["conns_saved"].append(conns_saved)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    if criterion == 't':
        for partitioning_step, data in partition_only_steps.items():
            ground_truth_lengths = data["ground_truth_lengths"]
            ground_truth_lengths[-1] = "{:,}".format(ground_truth_lengths[-1])
            conns_saved = data["conns_saved"]
            ax1.plot(ground_truth_lengths, conns_saved, marker='o', linestyle='-',
                     label=f"Splitting Factor = {partitioning_step}")
        ax1.set_xlabel('Number of Connections', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Connections Saved (%)', fontsize=14, fontweight='bold')
        # ax1.set_title('Average Times for Different Partitioning Steps', fontsize=16)
        legend1 = ax1.legend(loc='lower left')
        for text in legend1.get_texts():
            text.set_fontweight('bold')
            text.set_fontsize(12)
        ax1.set_ylim(0.5, 1)
    else:
        # Plot for average times
        for partitioning_step, data in partition_only_steps.items():
            # if partitioning_step in [2, 10]:
            ground_truth_lengths = data["ground_truth_lengths"]
            ground_truth_lengths[-1] = "{:,}".format(ground_truth_lengths[-1])
            average_times = data["average_times"]
            ax1.plot(ground_truth_lengths, average_times, marker='o', linestyle='-', linewidth=2,
                     label=f"Splitting Factor = {partitioning_step}")
        #ax1.plot(ground_truth_lengths, min_average_times, marker='*', markersize=10,
        #         linestyle='-', color='red', linewidth=2,
        #         label="N replica (parallel evaluation)")
        #ax1.plot(ground_truth_lengths, max_average_times, marker='o', markersize=8,
        #         linestyle='-', color='blue', linewidth=2,
        #         label="One replica (serial evaluation)")
        # Set labels and title for the first plot
        ax1.set_xlabel('Number of Connections', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Time (Mins)', fontsize=14, fontweight='bold')
        # ax1.set_title('Average Times for Different Partitioning Steps', fontsize=16)
        legend1 = ax1.legend(loc='upper left')
        for text in legend1.get_texts():
            text.set_fontweight('bold')
            text.set_fontsize(12)
        if criterion == 'cs':
            ax1.set_ylim(0, 20)
        else:
            ax1.set_ylim(0, 170)

    # Plot for total_vms_spawned
    if criterion == 'r':
        for partitioning_step, data in partition_only_steps.items():
            ground_truth_lengths = data["ground_truth_lengths"]
            conns_saved = data["conns_saved"]
            ax2.plot(ground_truth_lengths, conns_saved, marker='o', linestyle='-',
                     label=f"Splitting Factor = {partitioning_step}")
        ax2.set_xlabel('Number of Connections', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Connections Saved (%)', fontsize=14, fontweight='bold')
        # ax1.set_title('Average Times for Different Partitioning Steps', fontsize=16)
        legend1 = ax2.legend(loc='lower left')
        for text in legend1.get_texts():
            text.set_fontweight('bold')
            text.set_fontsize(12)
        ax2.set_ylim(0.5, 1)
    else:
        for partitioning_step, data in partition_only_steps.items():
            # if partitioning_step in [2, 10]:
            ground_truth_lengths = data["ground_truth_lengths"]
            total_vms = data["total_vms_spawned"]
            ax2.plot(ground_truth_lengths, total_vms, marker='o', linestyle='-', linewidth=2,
                     label=f"Splitting Factor = {partitioning_step}")
        #ax2.plot(ground_truth_lengths, min_total_vms_spawned, marker='*', markersize=10,
        #         linestyle='-', color='red', linewidth=2,
        #         label="N replica (parallel evaluation)")
        #ax2.plot(ground_truth_lengths, max_total_vms_spawned, marker='o', markersize=8,
        #         linestyle='-', color='blue', linewidth=2,
        #         label="One replica (serial evaluation)")
        # Set labels and title for the second plot
        ax2.set_xlabel('Number of Connections', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Replicas spawned', fontsize=14, fontweight='bold')
        # ax2.set_title('Total VMs Spawned for Different Partitioning Steps', fontsize=16)
        legend2 = ax2.legend(loc='upper left')
        for text in legend2.get_texts():
            text.set_fontweight('bold')
            text.set_fontsize(12)
        ax2.set_ylim(0, 2250)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Show the plots
    plt.show(block=True)


def plot_connection_distribution(file_path, GT_size, pol_rate, part_step, start_epoch=4, end_epoch=8):
    total_connections = 0

    with open(file_path, 'r') as file:
        inside_data = False
        end_of_file = False
        current_data = ""
        data_sets = []
        correct_data = f"---------- GroundTruth Length = {GT_size}, " \
                       f"Pollution factor = {pol_rate}, Partition step = {part_step} ----------"
        for line in file:
            if re.search(correct_data, line):
                match = re.search(r'\b\d+\b', correct_data)
                total_connections = int(match.group())
                inside_data = True
            elif inside_data and (re.search(r'---------- GroundTruth Length = \d+, '
                                            r'Pollution factor = \d+, Partition step = \d+ ----------',
                                            line) or not line.strip()):
                inside_data = False
                end_of_file = True
                data_sets.append(current_data.strip())
                current_data = ""
            elif inside_data:
                current_data += line
        if not end_of_file:
            data_sets.append(current_data.strip())
        data_line = [line for line in data_sets[0].split('\n') if "Average connections saved per epoch:" in line][0]
        average_connections_saved_per_epoch = [float(val) for val in data_line.split('[')[1].split(']')[0].split(',')]
        pollution_rate_per_node = []
        lines = data_sets[0].split('\n')
        partitions_per_epoch = []
        for line in lines:
            if "Average pollution rate per node in epoch" in line:
                parts = line.split(': [')
                values = [float(val) for val in parts[1].split(']')[0].split(',')]
                partitions_per_epoch.append(len(values))
                pollution_rate_per_node.append(values)

    # Extract data for the specified epochs
    average_connections_saved_per_epoch = average_connections_saved_per_epoch[start_epoch - 1:end_epoch]
    pollution_rate_per_node = pollution_rate_per_node[start_epoch - 1:end_epoch]

    # Calculate the connections saved in each epoch
    connections_saved_per_epoch = [average_connections_saved_per_epoch[0]] + [
        average_connections_saved_per_epoch[i] - average_connections_saved_per_epoch[i - 1] for i in
        range(1, len(average_connections_saved_per_epoch))]

    # Plotting
    plt.figure(figsize=(12, 8))

    # Set the margin values
    node_margin = 0.004  # Decrease the node margin

    # Plot healthy connections and malicious connections based on the new total
    for i, epoch_data in enumerate(pollution_rate_per_node):
        saved_connections = sum(connections_saved_per_epoch[:i + 1])
        new_total = total_connections - saved_connections

        # Calculate healthy connections for each node
        healthy_connections = np.array([new_total / len(epoch_data)] * len(epoch_data))

        # Plot healthy connections
        for j, healthy_value in enumerate(healthy_connections):
            plt.vlines(x=start_epoch + i + j * node_margin, ymin=0, ymax=healthy_value, color='green', linewidth=4,
                       alpha=0.7)

        # Calculate malicious connections for each node
        malicious_connections_new_total = np.array(epoch_data) * (new_total / len(epoch_data) / 100)

        # Plot malicious connections
        for j, malicious_value in enumerate(malicious_connections_new_total):
            plt.vlines(x=start_epoch + i + j * node_margin, ymin=0, ymax=malicious_value, color='red', linewidth=4,
                       alpha=0.7)

    plt.yticks(fontsize=14)
    plt.xlabel('Epoch Number', fontsize=14)
    plt.ylabel('Number of Connections', fontsize=14)

    plt.xticks(range(start_epoch, len(connections_saved_per_epoch) + start_epoch),
               labels=[f'Epoch {epoch} ({partitions_per_epoch[epoch - 1]})' for epoch in
                       range(start_epoch,
                             len(connections_saved_per_epoch)
                             + start_epoch)], fontsize=14)

    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_epoch_analytics(file_path, GT_size, pol_rate, part_step, epoch_num):
    key = ''

    with open(file_path, 'r') as file:
        data = file.read()
    if "partitions_only_epoch_analytics" in file_path:
        key = f"---------- GroundTruth Length = {GT_size}, Pollution factor = {pol_rate}, Partition step = {part_step} ----------"
    elif "distr_based_partitions_only_epoch_analytics" in file_path:
        key = f"---------- GroundTruth Length = {GT_size}, Pollution factor = {pol_rate} ----------"

    start_index = data.find(key)

    if start_index == -1:
        print("Data not found for the specified GroundTruth Length and Pollution factor.")
        return None

    # Find the start and end of the block for the specified epoch_num
    epoch_start_str = f"Average pollution rate per node in epoch {epoch_num}:"
    epoch_start_index = data.find(epoch_start_str, start_index)
    epoch_end_index = data.find(f"Average pollution rate per node in epoch {epoch_num + 1}:", epoch_start_index)

    if epoch_start_index == -1 or epoch_end_index == -1:
        print(f"Data not found for epoch {epoch_num}.")
        return None

    epoch_data = data[epoch_start_index + len(epoch_start_str):epoch_end_index].strip()
    # Remove brackets and split by ','
    epoch_data_cleaned = epoch_data.replace('[', '').replace(']', '')
    pollution_rates = list(map(float, epoch_data_cleaned.split(', ')))
    nodes = [f"{i + 1}" for i in range(len(pollution_rates))]
    plt.bar(nodes, pollution_rates, color='red')
    # plt.title(f'Pollution Rates for Epoch {epoch_num}')
    plt.xlabel('Nodes')
    plt.ylabel('Pollution Rate (%)')
    plt.xticks([19, 39, 59, 79, 99, 119, 139, 159, 179, 197],
               [20, 40, 60, 80, 100, 120, 140, 160, 180, 198])  # Set specific xticks
    plt.xlim(0, 198)
    plt.show()

    return pollution_rates


def plot_bathtub(file_path, GT_size, pol_rate, part_step1, part_step2):
    with open(file_path, 'r') as file:
        inside_data = False
        end_of_file = False
        current_data = ""
        data_sets = []
        total_connections = 0

        correct_data = f"---------- GroundTruth Length = {GT_size}, " \
                       f"Pollution factor = {pol_rate}, Partition step = {part_step1} ----------"
        correct_data2 = f"---------- GroundTruth Length = {GT_size}, " \
                        f"Pollution factor = {pol_rate}, Partition step = {part_step2} ----------"

        for line in file:
            if re.search(correct_data, line) or re.search(correct_data2, line):
                match = re.search(r'\b\d+\b', line)
                total_connections = int(match.group())
                inside_data = True
            elif inside_data and (re.search(r'---------- GroundTruth Length = \d+, '
                                            r'Pollution factor = \d+, Partition step = \d+ ----------',
                                            line) or not line.strip()):
                inside_data = False
                end_of_file = True
                data_sets.append(current_data.strip())
                current_data = ""
            elif inside_data:
                current_data += line
        if not end_of_file:
            data_sets.append(current_data.strip())

        # Extract data for the first pollution factor
        data_line1 = [line for line in data_sets[0].split('\n') if "Average connections saved per epoch:" in line][0]
        average_connections_saved_per_epoch1 = [float(val) / total_connections for val in
                                                data_line1.split('[')[1].split(']')[0].split(',')]
        data_line1 = [line for line in data_sets[0].split('\n') if "Average time elapsed per epoch:" in line][0]
        average_time_elapsed_per_epoch1 = [float(val) / 60000 for val in
                                           data_line1.split('[')[1].split(']')[0].split(',')]
        average_time_elapsed_per_epoch1 = list(accumulate(average_time_elapsed_per_epoch1))

        # Extract data for the second pollution factor
        data_line2 = [line for line in data_sets[1].split('\n') if "Average connections saved per epoch:" in line][0]
        average_connections_saved_per_epoch2 = [float(val) / total_connections for val in
                                                data_line2.split('[')[1].split(']')[0].split(',')]
        data_line2 = [line for line in data_sets[1].split('\n') if "Average time elapsed per epoch:" in line][0]
        average_time_elapsed_per_epoch2 = [float(val) / 60000 for val in
                                           data_line2.split('[')[1].split(']')[0].split(',')]
        average_time_elapsed_per_epoch2 = list(accumulate(average_time_elapsed_per_epoch2))

        # Plotting
        plt.figure(figsize=(12, 8))

        plt.plot(average_connections_saved_per_epoch1, average_time_elapsed_per_epoch1,
                 marker='o', linestyle='-', color='b', label=f'Partitioning Factor =  {part_step1}', markersize=8,
                 linewidth=2)

        plt.plot(average_connections_saved_per_epoch2, average_time_elapsed_per_epoch2,
                 marker='*', linestyle='-', color='r', label=f'Partitioning Factor = {part_step2}', markersize=10,
                 linewidth=2)

        plt.ylabel('Time elapsed (Mins)', fontsize=14, fontweight='bold')
        plt.xlabel('Average Connections Saved (%)', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.ylim(min(min(average_time_elapsed_per_epoch1), min(average_time_elapsed_per_epoch2)),
                 max(max(average_time_elapsed_per_epoch1), max(average_time_elapsed_per_epoch2)) + 1)

        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=14)

        # Fix y-axis ticks to standard intervals of 5
        y_ticks_interval = 5
        y_ticks = list(range(0, int(max(max(average_time_elapsed_per_epoch1), max(average_time_elapsed_per_epoch2)))
                             + y_ticks_interval, y_ticks_interval))
        plt.yticks(y_ticks, fontsize=14)

        legend2 = plt.legend(loc='upper left')
        for text in legend2.get_texts():
            text.set_fontweight('bold')
            text.set_fontsize(12)

        plt.show()

    return total_connections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot average times and connection distribution from simulation results.")

    parser.add_argument("--pol_factor", type=str, help="The pollution factor for which the time-length and Vm-length "
                                                       "plots.")
    parser.add_argument("--criterion", type=str, help="The criterion applied. Allowed values:"
                                                      " time (t), resources (r), connections saved (cs), none (n).")
    parser.add_argument("--strategy", type=str, help="The strategy for connections distribution plot. Allowed values:"
                                                     "partitions only (P), partitions only multiplication (Pm), "
                                                     "partitions only division (Pd), partitions only subtraction (Ps), "
                                                     "partitions only addition (Pa), Connections-size based (Pc).")
    parser.add_argument("--s_epoch", type=int, help="The starting epoch for connections distribution plot. "
                                                    "Ensure that the epoch number is valid.")
    parser.add_argument("--e_epoch", type=int, help="The ending epoch for connections distribution plot. "
                                                    "Ensure that the epoch number is valid.")
    parser.add_argument("--gt_len", type=int, help="The ground truth length for connections distribution plot.")
    parser.add_argument("--pf", type=int, help="The pollution factor for connections distribution plot.")
    parser.add_argument("--ps", type=int, help="The partition step for connections distribution plot.")
    parser.add_argument("--ez", type=int, help="The epoch that for zooming in.")

    args = parser.parse_args()
    if args.criterion == 't':
        plot_average_times(args.criterion, f'simulation_res/N_replicas/N_replicas_{args.pol_factor}.txt',
                           f'simulation_res/One_replica/One_replica_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only/parallel/time_crit/partitions_only_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_exp/parallel/time_crit/partitions_only_exp_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_rev_exp/parallel/time_crit/partitions_only_rev_exp_{args.pol_factor}.txt',
                           f'simulation_res/partionions_only_add/parallel/time_crit/partitions_only_add_{args.pol_factor}.txt',
                           f'simulation_res/partionions_only_sub/parallel/time_crit/partitions_only_sub_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_dynamic/parallel/time_crit/distr_based_partitions_only_{args.pol_factor}.txt')
    elif args.criterion == 'r':
        plot_average_times(args.criterion, f'simulation_res/N_replicas/N_replicas_{args.pol_factor}.txt',
                           f'simulation_res/One_replica/One_replica_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only/parallel/vms_crit/partitions_only_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_exp/parallel/vms_crit/partitions_only_exp_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_rev_exp/parallel/vms_crit/partitions_only_rev_exp_{args.pol_factor}.txt',
                           f'simulation_res/partionions_only_add/parallel/vms_crit/partitions_only_add_{args.pol_factor}.txt',
                           f'simulation_res/partionions_only_sub/parallel/vms_crit/partitions_only_sub_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_dynamic/parallel/vms_crit/distr_based_partitions_only_{args.pol_factor}.txt')
    elif args.criterion == 'cs':
        plot_average_times(args.criterion, f'simulation_res/N_replicas/N_replicas_{args.pol_factor}.txt',
                           f'simulation_res/One_replica/One_replica_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only/parallel/conns_saved_crit/partitions_only_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_exp/parallel/conns_saved_crit/partitions_only_exp_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_rev_exp/parallel/conns_saved_crit/partitions_only_rev_exp_{args.pol_factor}.txt',
                           f'simulation_res/partionions_only_add/parallel/conns_saved_crit/partitions_only_add_{args.pol_factor}.txt',
                           f'simulation_res/partionions_only_sub/parallel/conns_saved_crit/partitions_only_sub_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_dynamic/parallel/conns_saved_crit/distr_based_partitions_only_{args.pol_factor}.txt')
    elif args.criterion == 'n':
        plot_average_times(args.criterion, f'simulation_res/N_replicas/N_replicas_{args.pol_factor}.txt',
                           f'simulation_res/One_replica/One_replica_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only/parallel/without_crit/partitions_only_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_exp/parallel/without_crit/partitions_only_exp_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_rev_exp/parallel/without_crit/partitions_only_rev_exp_{args.pol_factor}.txt',
                           f'simulation_res/partionions_only_add/parallel/without_crit/partitions_only_add_{args.pol_factor}.txt',
                           f'simulation_res/partionions_only_sub/parallel/without_crit/partitions_only_sub_{args.pol_factor}.txt',
                           f'simulation_res/partitions_only_dynamic/parallel/without_crit/distr_based_partitions_only_{args.pol_factor}.txt')

    if args.strategy == 'P':
        plot_connection_distribution(
            'simulation_res/partitions_only/parallel/without_crit/partitions_only_epoch_analytics.txt',
            args.gt_len, args.pf, args.ps, args.s_epoch, args.e_epoch)
        plot_bathtub('simulation_res/partitions_only/parallel/without_crit/partitions_only_epoch_analytics.txt',
                     args.gt_len, args.pf, args.ps, 10)
        plot_epoch_analytics('simulation_res/partitions_only/parallel/without_crit/partitions_only_epoch_analytics.txt',
                             args.gt_len, args.pf, args.ps, args.ez)
    else:
        exit(0)
