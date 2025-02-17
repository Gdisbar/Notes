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

from huggingface_hub import login

HF_USERNAME = ""
HF_TOKEN = ""

try:
  login(token=HF_TOKEN)
except ValueError:
  login(username=HF_USERNAME, token=HF_TOKEN)


from huggingface_hub import HfApi

username = "pritam3355"
MODEL_NAME = quant_path = "Mistral-7B-AWQ-4bit"

api = HfApi(token=hf_tokens)

api.create_repo(
    repo_id = f"{username}/{MODEL_NAME}",
    repo_type="model"
)

api.upload_folder(
    repo_id = f"{username}/{MODEL_NAME}",
    folder_path = "/kaggle/working/Mistral-7B-AWQ-4bit"
)


!pip install -q colab-xterm
%load_ext colabxterm

%xterm -> inside xterm curl https://ollama.ai/install.sh | sh
ollama serve &  ollama pull llama --> start service
!ollama list
!ollama pull llama



# # Start the ZooKeeper service
# bin/zookeeper-server-start.sh config/zookeeper.properties
# # Start the Kafka broker service
# bin/kafka-server-start.sh config/server.properties
# # create topic to store events
# bin/kafka-topics.sh --create --topic testkafka-events --bootstrap-server localhost:9092
# bin/kafka-topics.sh --describe --topic testkafka-events --bootstrap-server localhost:9092
# # producer
# bin/kafka-console-producer.sh --topic testkafka-events --bootstrap-server localhost:9092
# # consumer
# bin/kafka-console-consumer.sh --topic testkafka-events --from-beginning --bootstrap-server localhost:9092
# # delete any data of your local Kafka environment including any events
# rm -rf /tmp/kafka-logs /tmp/zookeeper /tmp/kraft-combined-logs
# # alternatively to create topic
# kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic monthly-report

