# Just a big number for minimum calculation initialization
BIG = 10000


# Networking
PORT = 4999
HOST = "0.0.0.0"

# AprilTags
ROBOT_TAG_ID = 6
BASKET_TAG_ID = 0
## ROBOT: 
ROBOT_TAG_SIZE_MM = 96
## BASEKT: 
BASKET_TAG_SIZE_MM = 125

# Vision
CIRCULARITY_THRESH = 0.65
MIN_AERA = 150
MAX_AREA = 1500
ROBOT_SIZE_X = 200

# Coordinate estimation
ROLL_AVG_N = 10
JUMP_THRESHOLD_MM = 200 # Ignore the jump threshold logic
PERSISTENCE = ROLL_AVG_N * 2
MAX_INVALIDATE = 2