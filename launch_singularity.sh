#!/bin/bash

MODELS_SOURCE_DIR=/home/ychiba/link/ninkan-work/work/NTT/pitarie/20241001_tsuzumi7B-v1.2/models # host
MODELS_TARGET_DIR=/models
WORK_SOURCE_DIR=work
WORK_TARGET_DIR=/work

CONTAINER_NAME=tsuzumi-7B-v1.2

singularity exec --nv \
	    -B ${MODELS_SOURCE_DIR}:${MODELS_TARGET_DIR} \
	    -B ${WORK_SOURCE_DIR}:${WORK_TARGET_DIR} \
	    ${CONTAINER_NAME} /bin/bash

#singularity exec --nv \
#	    -B ${MODELS_SOURCE_DIR}:${MODELS_TARGET_DIR} \
#	    -B ./data:/data \
#	    -B ./experiments:/experiments \
#	    -B ./finetune_scripts:/finetune_scripts \
#	    -B ./.cache:/.cache \
#	    ${CONTAINER_NAME} ${SHELL}
