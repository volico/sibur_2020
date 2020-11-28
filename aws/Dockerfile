FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

copy modeling.py /opt/ml/train.py

COPY pl_framework.py /opt/ml/pl_framework.py
COPY NN.py /opt/ml/NN.py
COPY model_selection.py /opt/ml/model_selection.py
COPY load_data.py /opt/ml/load_data.py
COPY cv.py /opt/ml/cv.py
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "/opt/ml/train.py"]