CONTAINER_NAME="tsuzumi-7B-v1.2"
#singularity build --fakeroot --sandbox $CONTAINER_NAME tsuzumi.def
singularity build --fakeroot $CONTAINER_NAME tsuzumi.def
