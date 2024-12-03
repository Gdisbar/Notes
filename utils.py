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

from google.colab import userdata
import os
# Defined in the secrets tab in Google Colab
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')


!pip install -q colab-xterm
%load_ext colabxterm

%xterm -> inside xterm curl https://ollama.ai/install.sh | sh
ollama serve &  ollama pull llama --> start service
!ollama list
!ollama pull llama
