PIPELINE_CONFIG_PATH=/home/ubuntu/raccoon_project/pipeline.config
MODEL_DIR=/home/ubuntu/raccoon_project/models/model
python /home/ubuntu/git/tensorflow/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr

