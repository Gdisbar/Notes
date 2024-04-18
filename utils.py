!rm -r ~/.kaggle
!mkdir ~/.kaggle
!mv ./kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d wobotintelligence/face-mask-detection-dataset
!kaggle datasets download -c wobotintelligence/face-mask-detection-dataset

import zipfile
zip_ref = zipfile.ZipFile('face-mask-detection-dataset.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()


from google.colab import drive
drive.mount('/content/drive')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

%matplotlib inline


!pip install -q python-dotenv

import os
from dotenv import load_dotenv
load_dotenv()
