DPU_CONFIG=$1

ARCH_FILE="./arch_files/${DPU_CONFIG}.json"
OUTPUT_DIR="./outputs/${DPU_CONFIG}/"


vai_c_tensorflow2 \
    --model      ./quantized_model.h5 \
    --arch       $ARCH_FILE \
    --output_dir $OUTPUT_DIR \
    --net_name   $DPU_CONFIG
