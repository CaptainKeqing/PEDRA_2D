text = """absl-py==0.8.1
airsim==1.2.4
astor==0.8.0
bleach==1.5.0
certifi==2018.8.24
cycler==0.10.0
decorator 4.4.2
dotmap==1.3.8
enum34==1.1.6
gast 0.2.2
google-pasta 0.1.7
grpcio==1.24.1
h5py==2.10.0
html5lib==0.9999999
imageio==2.8.0
joblib==1.1.0
keras-applications==1.0.8
keras-preprocessing==1.1.0
kiwisolver==1.1.0
markdown==3.1.1
matplotlib==3.0.3
msgpack-python==0.5.6
msgpack-rpc-python==0.4.1
networkx==2.4
numpy==1.16.0
nvidia-ml-py3==7.352.0
opencv-python==4.1.1.26==
opt-einsum==3.1.0==
pillow==6.2.0
pip==21.2.2
protobuf 3.10.0
psutil==5.6.3
pygame==1.9.6
pyparsing==2.4.2
python==3.6.13
python-dateutil==2.8.0
pywavelets==1.1.1
scikit-image==0.15.0
scikit-learn==0.24.2
scipy==1.4.1
setuptools==58.0.4
six==1.12.0
sklearn==0.0==
sqlite==3.38.0
tensorboard==1.12.0
tensorflow==1.4.0
tensorflow-estimator==2.0.0
tensorflow-tensorboard==0.4.0
termcolor==1.1.0
threadpoolctl==3.1.0
tornado==4.5.3
vc==14.2
vs2015_runtime==14.27.29016
werkzeug==0.16.0
wheel==0.37.1
wincertstore==0.2
wrapt==1.11.2

Process finished with exit code 0
"""

text2 = text.replace(' ', '==')
print(text2)
