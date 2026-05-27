import threading
import config
from location import Location

# Global state
output_frame = None
lock = threading.Lock()
fps = 0
resolution = (0, 0)
rpi_name = ""
online = False
basket_loc = Location(config.ROLL_AVG_N, jump_threshold = config.JUMP_THRESHOLD_MM, persistence_threshold = config.PERSISTENCE)
garbage_loc = Location(config.ROLL_AVG_N, jump_threshold = config.JUMP_THRESHOLD_MM, persistence_threshold = config.PERSISTENCE)
garbage_id = 0


basket_i = 0
garbage_i = 0