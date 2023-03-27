import logging
import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.crazyflie.syncLogger import SyncLogger

uri = 'radio://0/90/2M/E7E7E7E703'

# Only output errors from the logging framework
# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)
received = 0
def log_stab_callback(timestamp, data, logconf):
    global received
    print('[%d][%s]: %s' % (timestamp, logconf.name, data))
    received = 1
    # sys.stdout.write("\r now is :{0}".format(data['stateEstimate.vz']))
    # sys.stdout.flush()
    # print()
def simple_log_async(scf, logconf):
    start_time = time.time()
    cf = scf.cf
    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(log_stab_callback)
    logconf.start()
    while received == 0:
        time.sleep(10)
    logconf.stop()
    print("time is :{0}".format(time.time() - start_time))

(...)

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    # lg_stab.add_variable('stateEstimateZ.z', 'float')
    # lg_stab.add_variable('stateEstimate.ax', 'float')
    # lg_stab.add_variable('stateEstimate.ay', 'float')
    # lg_stab.add_variable('stateEstimate.vz', 'float')
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

        simple_log_async(scf, lg_stab)

