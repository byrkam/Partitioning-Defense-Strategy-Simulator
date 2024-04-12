import argparse
import configparser
import json


def initialize_simulation_config(filename, args):
    config = configparser.ConfigParser()
    config.read(filename)

    # Update values based on command line arguments
    if args.connections:
        # Convert the list of strings to integers
        range_values = list(map(int, args.connections))
        connection_values = list(range(range_values[0], range_values[1] + 1, range_values[2]))
        config['Topology']['NumberofConnections'] = json.dumps(connection_values)
    if args.partition_step:
        config['Topology']['numberofnodes'] = json.dumps(args.partition_step)
    if args.pollution_factor:
        config['Topology']['polutionfactor'] = str(args.pollution_factor)
    if args.percent_migrate:
        config['Defense']['PercentangeofConnsToMigrate'] = str(args.percent_migrate)
    if args.max_time_global:
        config['StopCriteria']['MaxTimeElapsedGlobal'] = str(args.max_time_global)
    if args.max_time_criterion:
        config['StopCriteria']['applymaxtimecriterion'] = str(args.max_time_criterion)
    if args.connections_saved:
        config['StopCriteria']['minconnectionssafeglobal'] = str(args.connections_saved)
    if args.connections_safe_criterion:
        config['StopCriteria']['applyconnectionssafecriterion'] = str(args.connections_safe_criterion)
    if args.max_cost:
        config['StopCriteria']['maxcostglobal'] = str(args.max_cost)
    if args.max_cost_criterion:
        config['StopCriteria']['applymaxcostcriterion'] = str(args.max_cost_criterion)
    if args.GT_distr:
        config['Simulation']['groundtruthgentype'] = str(args.GT_distr)
    # Add more updates based on command line arguments

    # Save the changes back to the file
    with open(filename, 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize simulation parameters.')

    parser.add_argument('--C', dest='connections', type=int, nargs='+', help='List of connection values as separate '
                                                                             'arguments')
    parser.add_argument('--Ps', dest='partition_step', type=int, nargs='+', help='Partition step')
    parser.add_argument('--Pf', dest='pollution_factor', type=int, nargs='+', help='Pollution factor of the ground '
                                                                                   'truth')
    parser.add_argument('--M', dest='percent_migrate', type=int, nargs='+', help='Percentage of connections to migrate')
    parser.add_argument('--mT', dest='max_time_global', type=int, help='Stopping Criterion1: Max time elapsed globally')
    parser.add_argument('--mTe', dest='max_time_criterion', type=str, help='Enable Criterion1. Allowed values YES or NO')
    parser.add_argument('--Cs', dest='connections_saved', type=int, help='Stopping Criterion2: Percentage of '
                                                                         'Connections saved')
    parser.add_argument('--Cse', dest='connections_safe_criterion', type=str, help='Enable Criterion2. Allowed values YES or NO')
    parser.add_argument('--mC', dest='max_cost', type=int, help='Stopping Criterion3: Total VMs spawned')
    parser.add_argument('--mCe', dest='max_cost_criterion', type=str, help='Enable Criterion3. Allowed values YES or NO')
    parser.add_argument('--GT', dest='GT_distr', type=str, help='Ground truth distribution. Allowed values: "RANDOM", "NORMAL"')
    # Add more command line arguments as needed

    args = parser.parse_args()

    filename = "params.ini"
    initialize_simulation_config(filename, args)
