
import numpy as np
from PIL import Image 
import datetime
import os

import tritonclient.http as httpclient

#Run the docker with the following commands;
#docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/logs:/tmp/ --mount type=bind,source=$(pwd)/models,target=/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models --metrics-config summary_latencies=true --trace-config triton,file=/tmp/trace.json --trace-config triton,log-frequency=50 --trace-config rate=1 --trace-config level=TIMESTAMPS --trace-config count=100


TRITON_SERVER_URL = 'localhost:8000'
MODEL_NAME = 'mimo-unet'

input_image_path = 'sample_input_imgs/000023.png'
head, input_filename = os.path.split(input_image_path)

demo_image_input = Image.open(input_image_path).convert("RGB")  

#mimo-unet tranpose image to 
width, height = demo_image_input.size
new_width = width + (8 - width % 8)
new_height = height + ( 8 - height % 8)

result = Image.new(demo_image_input.mode, (new_width, new_height), (0, 0, 0))
result.paste(demo_image_input, (0,0)) #default loads in h,w,c format. 

I = np.asarray(result) 
I = (I/255.0).astype(np.float32)

#transpose to 
input_img_tensor_numpy = np.expand_dims(I.transpose(2, 0, 1), axis=0) #convert to (b,c,h,w) format

print('input_img_tensor_numpy: ', input_img_tensor_numpy.shape)

#create inference URL
request_uri = f"v2/models/{MODEL_NAME}/infer"
triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

#create inference input. must match same input shape as config.pbtxt
inp = httpclient.InferInput('input1', input_img_tensor_numpy.shape, 'FP32')
inp.set_data_from_numpy(input_img_tensor_numpy,  binary_data=True)

start_time = datetime.datetime.now()
print("Sending Request")

try:
    infer_response = triton_client.infer(
        model_name="mimo-unet", inputs=[inp]
    )

    #post processing
    output_image = infer_response.as_numpy("output3")
    output_image += 0.5/255
    pred_clip = np.clip(output_image, 0, 1) * 255
    save_name = f'output/{input_filename}'
    pred = Image.fromarray(np.squeeze(pred_clip, axis=0).transpose(1, 2, 0).astype('uint8'))
    pred.save(save_name)

#except httpclient.InferenceServerException as e:
except httpclient.InferenceServerException as e:
    print(f"[Error {e.status()} ] {e.message()} ")

except Exception as e:
    print(e)

elapsed = datetime.datetime.now() - start_time
print("Received Response in {0} seconds".format(elapsed.total_seconds()))