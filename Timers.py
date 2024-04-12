import random
import Params

total_elapsed_time = 0


def getTotalElapsedTime():
    global total_elapsed_time

    return total_elapsed_time


def increaseTotalElapsedTime(time):
    global total_elapsed_time

    total_elapsed_time = total_elapsed_time + time


def getRandomFreezeTime(iter=1):
    global total_elapsed_time

    total = 0
    for i in range(iter):
        total = total + random.gauss(Params.MEANFREEZETIME, Params.STDFREEZETIME)
    return total


def getRandomSpawnTime(iter=1):
    global total_elapsed_time

    total = 0
    for i in range(iter):
        total = total + random.gauss(Params.MEANSPAWNTIME, Params.STDSPAWNTIME)
    return total


def getRandomMigrationTime(iter=1):
    global total_elapsed_time

    total = 0
    for i in range(iter):
        total = total + random.gauss(Params.MEANMIGRATIONTIME, Params.STDMIGRATIONTIME)
    return total 
    

def getRandomRestoreTime(iter=1):
    global total_elapsed_time

    total = 0
    for i in range(iter):
        total = total + random.gauss(Params.MEANRESTORETIME, Params.STDRESTORETIME)
    return total 


def getRandomEvalTime(iter=1):
    global total_elapsed_time

    total = 0
    for i in range(iter):
        total = total + random.gauss(Params.MEANEVALTIME, Params.STDEVALTIME)
    return total
