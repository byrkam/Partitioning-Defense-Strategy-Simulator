import configparser
import ast
import numpy as np

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read('params.ini')

# Topology
NUMBEROFCONNECTIONS = ast.literal_eval(config["Topology"]["NumberofConnections"])
NUMBEROFNODES = ast.literal_eval(config["Topology"]["NumberofNodes"])
POLUTIONFACTOR = ast.literal_eval(config["Topology"]["PolutionFactor"])

TOTALMALICIOUSCONNECTIONS = []
for connections in NUMBEROFCONNECTIONS:
    MALICIOUSPERPOLLUTIONFACTOR = []
    TOTALMALICIOUSCONNECTIONS.append(MALICIOUSPERPOLLUTIONFACTOR)
    for pol_factor in POLUTIONFACTOR:
        MALICIOUSPERPOLLUTIONFACTOR.append(int(connections * pol_factor / 100))

# Defense
PERCENTAGEOFCONNSTOMIGRATE = ast.literal_eval(config["Defense"]["PercentangeofConnsToMigrate"])

# StopCriteria
MAXTIMEELAPSEDGLOBAL = float(config["StopCriteria"]["MaxTimeElapsedGlobal"])
MAXTIMECRITERION = str(config["StopCriteria"]["applymaxtimecriterion"])
MINCONNECTIONSSAFEGLOBALPC = float(config["StopCriteria"]["MinConnectionsSafeGlobal"]) / 100
CONNECTIONSSAFECRITERION = str(config["StopCriteria"]["applyconnectionssafecriterion"])
MAXCOSTGLOBAL = float(config["StopCriteria"]["MaxCostGlobal"])
MAXCOSTCRITERION = str(config["StopCriteria"]["applymaxcostcriterion"])

# Simulation
MAXSPAWNTIME = float(config["Simulation"]["MaxSpawnTime"])
MINSPAWNTIME = float(config["Simulation"]["MinSpawnTime"])
MEANSPAWNTIME = float(config["Simulation"]["MeanSpawnTime"])
STDSPAWNTIME = float(config["Simulation"]["StdSpawnTime"])
MAXFREEZETIME = float(config["Simulation"]["MaxFreezeTime"])
MINFREEZETIME = float(config["Simulation"]["MinFreezeTime"])
MEANFREEZETIME = float(config["Simulation"]["MeanFreezeTime"])
STDFREEZETIME = float(config["Simulation"]["StdFreezeTime"])
MAXMIGRATIONTIME = float(config["Simulation"]["MaxMigrationTime"])
MINMIGRATIONTIME = float(config["Simulation"]["MinMigrationTime"])
MEANMIGRATIONTIME = float(config["Simulation"]["MeanMigrationTime"])
STDMIGRATIONTIME = float(config["Simulation"]["StdMigrationTime"])
MAXRESTORETIME = float(config["Simulation"]["MaxRestoreTime"])
MINRESTORETIME = float(config["Simulation"]["MinRestoreTime"])
MEANRESTORETIME = float(config["Simulation"]["MeanRestoreTime"])
STDRESTORETIME = float(config["Simulation"]["StdRestoreTime"])
MAXEVALTIME = float(config["Simulation"]["MaxEvalTime"])
MINEVALTIME = float(config["Simulation"]["MinEvalTime"])
MEANEVALTIME = float(config["Simulation"]["MeanEvalTime"])
STDEVALTIME = float(config["Simulation"]["StdEvalTime"])
GROUNDTRUTHGENTYPE = config["Simulation"]["GroundTruthGenType"]
MEANGROUNDTRUTH = float(config["Simulation"]["meangroundtruth"])
STDGROUNDTRUTH = float(config["Simulation"]["stdgroundtruth"])

GROUNDTRUTH = []

if GROUNDTRUTHGENTYPE == "RANDOM":
    for i in range(len(NUMBEROFCONNECTIONS)):
        connection_count = NUMBEROFCONNECTIONS[i]
        malicious_counts = TOTALMALICIOUSCONNECTIONS[i]

        row = []
        for count in malicious_counts:
            malicious_indices = np.random.choice(connection_count, count, replace=False)
            row.extend(np.isin(range(1, connection_count + 1), malicious_indices).astype(int))
        GROUNDTRUTH.append(row)
    GROUNDTRUTH = [[inner_list[i:i + len(inner_list) // len(POLUTIONFACTOR)] for i in
                    range(0, len(inner_list), len(inner_list) // len(POLUTIONFACTOR))] for inner_list in GROUNDTRUTH]
elif GROUNDTRUTHGENTYPE == "NORMAL":
    for i in range(len(NUMBEROFCONNECTIONS)):
        connection_count = NUMBEROFCONNECTIONS[i]
        malicious_counts = TOTALMALICIOUSCONNECTIONS[i]
        std_devs = []
        row = []
        for index, count in enumerate(malicious_counts):
            mean_index = connection_count * MEANGROUNDTRUTH
            for mal_count in malicious_counts:
                std_devs.append(connection_count * STDGROUNDTRUTH)
            # Initialize an empty array for indexes
            indexes = np.array([])

            # Generate unique indexes until you have exactly equal to the malicious_counts
            while len(indexes) < count:
                # Generate random indexes from a normal distribution
                additional_indexes = np.round(
                    np.random.normal(mean_index, std_devs[index], size=count)).astype(int)

                # Ensure that generated indexes are within the valid range (0 to 999) and non-negative
                additional_indexes = np.clip(additional_indexes, 0, np.inf)

                # Make sure new indexes are unique
                additional_indexes = np.setdiff1d(additional_indexes, indexes)

                # Concatenate the new unique indexes with the existing ones
                indexes = np.concatenate((indexes, additional_indexes))
            indexes = indexes[:count]
            row.extend(np.isin(range(1, connection_count + 1), indexes).astype(int))
            GROUNDTRUTH.append(row)
            row = []

    GROUNDTRUTH = sorted(GROUNDTRUTH, key=len)

    # Group the original lists based on their lengths
    grouped_lists = {}
    for sublist in GROUNDTRUTH:
        sublist_len = len(sublist)
        if sublist_len not in grouped_lists:
            grouped_lists[sublist_len] = []
        grouped_lists[sublist_len].append(sublist)

    # Create the final structure
    GROUNDTRUTH = [grouped_lists[sublist_len] for sublist_len in grouped_lists]

NUMBEROFTESTS = int(config["Simulation"]["NumberofTests"])
