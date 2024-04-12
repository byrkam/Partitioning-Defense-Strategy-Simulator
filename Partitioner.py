import Timers


def split_list_into_sublists(input_list, num_sublists):
    # Calculate the size of each sublist
    sublist_size = len(input_list) // num_sublists
    # Handle any remaining elements
    remainder = len(input_list) % num_sublists

    # Initialize an empty list to store the sublists
    sublists = []

    # Initialize the start index for slicing
    start_idx = 0

    for i in range(num_sublists):
        # Calculate the end index for slicing
        end_idx = start_idx + sublist_size + (1 if i < remainder else 0)

        # Append the sublist to the list of sublists
        sublist = input_list[start_idx:end_idx]
        sublists.append(sublist)

        # Update the start index for the next iteration
        start_idx = end_idx

    sublists = [sublist for sublist in sublists if sublist]

    return sublists


def doPartitioning(original_partition, partition_step, criterion1, criterion2, criterion3, max_time_elapsed,
                   min_connections_saved,
                   max_number_of_nodes, total_connections_saved, strategy=1):
    partition_time = 0

    if criterion1 == "YES" and Timers.total_elapsed_time > max_time_elapsed:
        return original_partition, partition_time
    if criterion2 == "YES" and total_connections_saved > min_connections_saved:
        return original_partition, partition_time

    new_partition = []

    largest_node = max(original_partition, key=len)

    for node in original_partition:
        new_partition.extend(split_list_into_sublists(node, partition_step))
        if criterion3 == "YES" and len(new_partition) > max_number_of_nodes:
            return new_partition, partition_time

    if strategy == 0:
        num_nodes = 1
    else:
        num_nodes = partition_step * len(original_partition)

    Timers.increaseTotalElapsedTime(Timers.getRandomSpawnTime(num_nodes))
    Timers.increaseTotalElapsedTime(Timers.getRandomFreezeTime(len(largest_node)))
    Timers.increaseTotalElapsedTime(Timers.getRandomMigrationTime(len(largest_node)))
    Timers.increaseTotalElapsedTime(Timers.getRandomRestoreTime(len(largest_node)))

    partition_time = Timers.getRandomSpawnTime(num_nodes) + Timers.getRandomFreezeTime(len(largest_node)) + \
                     Timers.getRandomMigrationTime(len(largest_node)) + Timers.getRandomRestoreTime(len(largest_node))

    return new_partition, partition_time
