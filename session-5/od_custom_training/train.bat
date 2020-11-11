SET PIPELINE_CONFIG_PATH=C:/Users/markk/raccoon_project/models/pipeline.config
SET MODEL_DIR=C:/Users/markk/raccoon_project/models/model
python object_detection/model_main_tf2.py^
    --pipeline_config_path=%PIPELINE_CONFIG_PATH%^
    --model_dir=%MODEL_DIR%^
    --alsologtostderr

