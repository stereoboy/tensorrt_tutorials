XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

#        --device /dev/video0 \ # for webcam
#        --device /dev/bus/usb \ # for flir camera

docker run  --gpus device=1 -dt -it \
        --volume=$XSOCK:$XSOCK:rw \
        --volume=$XAUTH:$XAUTH:rw \
        --env="XAUTHORITY=${XAUTH}" \
        --env="DISPLAY=${DISPLAY}" \
        --env QT_X11_NO_MITSHM=1 \
        --env="NVIDIA_DRIVER_CAPABILITIES=all" \
        --device /dev/video0 \
        --device /dev/bus/usb \
        -v /home/jylee/work2:/work2 \
        -w /work2 \
        --name tf-trt6-con \
        --net=host \
   tf-trt6-spinnaker:latest

# options
#        --privileged \
