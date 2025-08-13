#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

# Allow X11 connections for Docker
xhost +local:docker

# Navigate to the docker directory
cd "$(dirname "$0")"

docker-compose down

# Determine mode based on CLI argument
if [[ "$1" == "-i" ]]; then
    echo "Running in interactive mode..."
    docker-compose up --build --remove-orphans
else
    echo "Running in detached mode..."
    docker-compose up --build --remove-orphans -d
fi