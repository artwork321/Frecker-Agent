# Default settings
DL_ENV=${1:-"pytorch"}
NOTEBOOK_PORT=${2:-8888}
VISDOM_PORT=${3:-8097}
LOCAL_IMAGE_NAME=alpha-zero-project

# Correct string comparison
if [ "$DL_ENV" == "pytorch" ]; then
    # Build the Docker image if it doesn't exist
    if [[ -z $(docker images -q ${LOCAL_IMAGE_NAME}) ]]; then
        echo "Docker image not found. Building..."
        docker build . -t ${LOCAL_IMAGE_NAME} -f ./docker/Dockerfile.pytorch
    fi

    echo "Starting container for Jupyter notebook..."

    # Run the container
    docker run -it --rm \
        --shm-size=8G \
        -v "$(pwd)":/workspace \
        -p ${NOTEBOOK_PORT}:8888 \
        -p ${VISDOM_PORT}:8097 \
        --name pytorch_notebook \
        ${LOCAL_IMAGE_NAME} \
        bash -c "jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"
else
    echo "Unknown DL_ENV: $DL_ENV"
    exit 1
fi
