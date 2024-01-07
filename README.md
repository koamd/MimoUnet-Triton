# MimoUnet-Triton

A triton deployment implementation of MimoUnet

The original Mimo-Unet model was obtained from:

[https://github.com/chosj95/MIMO-UNet](https://github.com/chosj95/MIMO-UNet)

The Mimo-Unet model has been converted to ONNX format

## 1\. Docker Setup for Triton Server

1.  In the /triton folder, the /models folder is the model repository folder and it contains the Mimo-Unet model and its config.pbtxt.
2.  In the /triton folder, the /logs folder is the location where our trace logs will be stored.
3.  Run the docker from the /triton folder using the following command :

```
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/logs:/tmp/ --mount type=bind,source=$(pwd)/models,target=/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models --metrics-config summary_latencies=true --trace-config triton,file=/tmp/trace.json --trace-config triton,log-frequency=50 --trace-config rate=1 --trace-config level=TIMESTAMPS --trace-config count=100
```

Triton inference server will be started on localhost:8000 (http)

## 2\. Triton Client

An sample client code is provided in [test_mimounet_tritonclient.py](test_mimounet_tritonclient.py)

To run the triton client, we need to install the dependencies required in requirements.txt

```
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the sample triton client code:

```
python test_mimounet_tritonclient.py
```

## 3\. Logs Analysis

We can use our util files to track the trace logs on the time used to track each inference call.

```
python utils/tracesummary.py triton/logs/trace.json.0
```

An example output would be:

```
File: triton/logs/trace.json.0
Summary for mimo-unet (1): trace count = 3
HTTP infer request (avg): 2430110.905333333us
        Receive (avg): 6959.4276666666665us
        Send (avg): 20.861us
        Overhead (avg): 100.13733333333333us

        Handler (avg): 2423030.479333333us
                Overhead (avg): 48.001333333333335us
                Queue (avg): 64.294us
                Compute (avg): 2422918.184us
                        Input (avg): 24.932666666666666us
                        Infer (avg): 2418610.7533333334us
                        Output (avg): 4282.498us
```
