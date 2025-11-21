#!/bin/bash

MLCOMMONS_REPO_PATH="$(dirname "$(dirname "$PWD")")"
MLCOMMONS_ALL_PATH="$(dirname "$MLCOMMONS_REPO_PATH")"

# Add any volume mounts here with the following syntax
# /path/to/src:/path/to/dir/in/container
MOUNTS=(
    $MLCOMMONS_ALL_PATH:$MLCOMMONS_ALL_PATH
    $MLCOMMONS_REPO_PATH:$MLCOMMONS_REPO_PATH
)

# Set up docker environment file for current user
rm -f .docker_env
echo "CI_BUILD_USER=`id -u -n`" >> .docker_env
echo "CI_BUILD_UID=`id -u`" >> .docker_env
echo "CI_BUILD_GROUP=`id -g -n`" >> .docker_env
echo "CI_BUILD_GID=`id -g`" >> .docker_env
cat .docker_env

# Build container
docker build --network=host . -t llm/gpubringup

# Build mount flags
declare -a MOUNT_FLAGS
for _mount in ${MOUNTS[@]}; do
    _split=($(echo $_mount | tr ':' '\n'));
    MOUNT_FLAGS+=("--mount type=bind,source=${_split[0]},target=${_split[1]}");
done

set -x
docker run -it --rm --gpus all --net=host --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH \
  --security-opt seccomp=unconfined \
  -w $PWD \
  --env-file `pwd`/.docker_env \
  ${MOUNT_FLAGS[*]} \
  llm/gpubringup \
  bash ./with_the_same_user
