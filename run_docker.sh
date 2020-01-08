XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run  --gpus=all -dt -it \
        --volume=$XSOCK:$XSOCK:rw \
        --volume=$XAUTH:$XAUTH:rw \
        --env="XAUTHORITY=${XAUTH}" \
        --env="DISPLAY=${DISPLAY}" \
        --env="NVIDIA_DRIVER_CAPABILITIES=all" \
        -v /home/jylee/work2:/work2 \
        -w /work2 \
        --privileged \
        --name tf-trt6-con \
   tf-trt6:latest
