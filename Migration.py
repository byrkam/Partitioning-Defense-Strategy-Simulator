import Timers


def migrate(node, num_conns_to_migrate, next_index):
    conns_to_migrate = node[next_index:next_index + num_conns_to_migrate]
    Timers.increaseTotalElapsedTime(Timers.getRandomFreezeTime(len(2*conns_to_migrate)))
    Timers.increaseTotalElapsedTime(Timers.getRandomMigrationTime(len(2*conns_to_migrate)))
    Timers.increaseTotalElapsedTime(Timers.getRandomRestoreTime(len(2*conns_to_migrate)))

    return conns_to_migrate





