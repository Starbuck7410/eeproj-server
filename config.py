# Just a big number for minimum calculation initialization
BIG = 10000


# Networking
PORT = 4999
HOST = "0.0.0.0"

# AprilTags
ROBOT_TAG_ID = 67
BASKET_TAG_ID = 69
## ROBOT: 
ROBOT_TAG_SIZE_MM = 61.2
## BASEKT: 
BASKET_TAG_SIZE_MM = 131.8

# Vision
CIRCULARITY_THRESH = 0.69
MIN_AERA = 300

# Coordinate estimation
JUMP_THRESHOLD_MM = BIG # Ignore the jump threshold logic
PERSISTENCE = 1 # ROLL_AVG_N // 3
ROLL_AVG_N = 10
