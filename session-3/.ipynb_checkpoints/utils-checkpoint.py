import os
import shutil
import zipfile
import numpy as np
import urllib.request
from tqdm import tqdm
import tensorflow as tf

def fix_cudnn_bug(): 
    # during training, tf will throw cudnn initialization error: failed to get convolution algos
    # the following codes somehow fix it
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
        
def download_data(url, target_filepath, force=False):
    # if not force download and directory already exists
    if target_filepath and not force:
        if os.path.exists(target_filepath): 
            print('target file already exists, skip download')
            return
    
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        filename, _ = urllib.request.urlretrieve(url, reporthook=t.update_to)
        print(filename)
        if target_filepath != None:
            os.rename(filename, target_filepath)

def prepare_data(data_path="./data", valid_size=0.2, seed=21, FORCED_DATA_REWRITE=False):
    
    url = 'https://sdaai-bucket.s3-ap-southeast-1.amazonaws.com/datasets/intel_emotions_dataset.zip'
    
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")

    if FORCED_DATA_REWRITE:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

    if not os.path.exists(data_path):
        if not os.path.exists("intel_emotions_dataset.zip"):
            download_data(url, 'intel_emotions_dataset.zip')
            #os.system("wget https://sdaaidata.s3-ap-southeast-1.amazonaws.com/datasets/intel_emotions_dataset.zip")
        
        zip_ref = zipfile.ZipFile("intel_emotions_dataset.zip", "r")
        zip_ref.extractall("data_temp")
        zip_ref.close()
        
        os.rename("data_temp", data_path)
        #shutil.rmtree("data_temp")
        
        os.mkdir(train_path)
        os.mkdir(valid_path)

        for category in ["Negative", "Positive"]:
            train_emo_path = os.path.join(train_path, category)
            valid_emo_path = os.path.join(valid_path, category)
            os.mkdir(train_emo_path)
            os.mkdir(valid_emo_path)

            categoty_list = np.array(os.listdir(os.path.join(data_path, category)))
            np.random.seed(seed)
            np.random.shuffle(categoty_list)
            
            train_list = categoty_list[int(len(categoty_list) * valid_size):]
            valid_list = categoty_list[:int(len(categoty_list) * valid_size)]
            
            for filename in train_list:
                os.rename(os.path.join(data_path, category, filename), 
                          os.path.join(train_emo_path, filename.replace(" ", "")))
                
            for filename in valid_list:
                os.rename(os.path.join(data_path, category, filename), 
                          os.path.join(valid_emo_path, filename.replace(" ", "")))
                
            shutil.rmtree(os.path.join(data_path, category))

    return train_path, valid_path


def download_trained_model_and_history(model_path):
    model_url = 'https://sdaai-bucket.s3-ap-southeast-1.amazonaws.com/pretrained-weights/iti107/session-3/baseline.model.h5'
    baseline_history_url = 'https://sdaai-bucket.s3-ap-southeast-1.amazonaws.com/pretrained-weights/iti107/session-3/baseline.history' 
    
    if not os.path.exists(model_path):
        download_data(model_url, model_path, force=False)
#         os.system("wget https://sdaaidata.s3-ap-southeast-1.amazonaws.com/pretrained-weights/iti107/session-3/baseline.model.h5")
        #os.rename('baseline.model.h5',model_path)
    if not os.path.exists('baseline.history'):
#         os.system("wget https://sdaaidata.s3-ap-southeast-1.amazonaws.com/pretrained-weights/iti107/session-3/baseline.history")
        download_data(baseline_history_url, 'baseline.history',force=False)


if __name__ == '__main__':
    #prepare_data('data', valid_size=0.2, FORCED_DATA_REWRITE=True)
    download_trained_model_and_history(os.path.join('models', 'baseline.model.h5'))

    #download_trained_model_and_history('models')

