from itertools import chain

output_file_name = "epoch_analytics.txt"


def epoch_analytics(groundTruth, config, time_per_epoch):
    malicious_conns_per_node = 0
    node_health = []
    connections_saved = len(groundTruth) - len(list(chain.from_iterable(config)))
    for index, node in enumerate(config):
        for connection in node:
            if connection == 1:
                malicious_conns_per_node += 1
        node_health.append(malicious_conns_per_node/len(node) * 100)
        malicious_conns_per_node = 0

    return connections_saved, len(config), node_health, time_per_epoch


def split_file_by_delimiter(input_file_path, output_file_base):
    with open(input_file_path, 'r') as file:
        data = file.read()
    if 'Partition step:' in data:
        delimiter_prefix = 'Distribution: NORMAL----------Pollution factor: '
        delimiter_suffix = '----------'
        sections = data.split(delimiter_prefix)
        prev_pollution_factor = None

        for section in sections[1:]:
            header, *content = section.split(delimiter_suffix)
            pollution_factor, _ = map(str.strip, header.split(', Partition step: '))
            if pollution_factor != prev_pollution_factor:
                output_file_name = f'{output_file_base}_{pollution_factor}.txt'
                prev_pollution_factor = pollution_factor
            with open(output_file_name, 'a') as output_file:
                output_file.write(delimiter_prefix + section)
    else:
        delimiter = 'Distribution: NORMAL----------Pollution factor: '
        sections = data.split(delimiter)
        for section in sections[1:]:
            pollution_factor = section.split('----------')[0].strip()
            output_file_name = f'{output_file_base}_{pollution_factor}.txt'
            with open(output_file_name, 'a') as output_file:
                output_file.write(delimiter + section)
