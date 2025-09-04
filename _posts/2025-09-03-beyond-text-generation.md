---
layout: post
title: "Enabling GeospatialVision transofrmers as first class citizens in vLLM"
author: "Christian Pinto (IBM Research Europe), Michele Gazzetti (IBM Research Europe)
image: /assets/logos/vllm-logo-text-light.png
---
## Introduction
Until recently, AI inference has largely revolved around one modality: text. From chatbots to summarizers to code generators, the focus has been on generating and serving language-based outputs. But the frontier is shifting. As models become increasingly multimodal—capable of producing images, audio, structured data, and more—the infrastructure for serving them must evolve too. One class of models of great interest is geospatial vision transformers, used for performing classification tasks on satellite images for detecting floods, burn scars, land use, etc. We at IBM Research have developed in collaboration with NASA a set of models for geospatial applications.

vLLM has become the de facto standard for high-throughput, low-latency model serving. Its performance, scalability, and ease of integration have made it the go-to choice for deploying large language models in production. However, serving any model that is not generating text was not possible, until now. 

In this article we introduce the set of changes required for making vLLM a generic framework for inferencing models with multi-modal input/output requirements.

In the specific, the main hurdles towards supporting geospatial vision transformer models in vLLM were:
- Enabling generation of multi-modal data
- Supporting pre-processing of input data to support tasks such as images segmentation
- Integrate geospatial vision transformer models in vLLM

## Meet IOProcessor: Flexible Input/Output Handling for Any Model

Pre-processing of input data and pre-processing of the model output is only partially possible in vLLM. Specifically, the preprocessing of input data into the format required by the model is possible via processors the transformers library. However, transformers processors usually support only default data types and do not deal with complex formats such as `geotiff` that enrich tiff files with georeferenced metadata. Also, on the output processing side vLLM support detokenization into text or the application of poolers to the model hidden states.

We have recently merged into the vLLM mainline a new framework called IOProcessor plugins. The IOProcessor framework allows developers to customize how model inputs and outputs are pre and post processed, all within the same vLLM serving instance. Whether your model returns a string, a JSON object, an image tensor, or a custom data structure, IOProcessor can translate it into the desired format before returning it to the client.

This means that you can serve non-text models (e.g., image generators, embedding models, structured output models) using the same vLLM infrastructure.
You can plug in custom logic to transform or enrich outputs such as decoding tokens into images, parsing structured outputs, or formatting responses for downstream systems.
You maintain a unified serving stack, reducing operational complexity and improving maintainability.

The IOProcessor framework unlocks a new level of flexibility for vLLM users. Instead of building and maintaining separate serving pipelines for each model type, you can now consolidate everything under one roof.

In the case of vision transformers, input images can be 

### Using vLLM IOProcessor plugins

Each IOProcessor plugin implements a pre-defined [IOProcessor interface](https://github.com/vllm-project/vllm/blob/main/vllm/plugins/io_processors/interface.py) and reside outside of the vLLM source code tree. At installation time each plugin registers one or more entrypoints in the `vllm.io_processor_plugins` group. This allows for vLLM to automatically discovery and load plugins at initialization time. A full plugin example for egospatial vision transformers is available [here](https://github.com/christian-pinto/prithvi_io_processor_plugin).

Using an IOProcessor plugin is as easy as just installing it in the same python environment with vLLM and add the `io-processor-plugin <plugin_name>` when starting the serving instance.

Once the serving instance is started, pre and post processing is automatically applied to the model input and output when serving the `encode` endpoint.
At this stage only pooling models are served with IOProcessor plugins, but in the future we expect other types of models to integrated too.

### Integrate geospatial vision transformer models in vLLM

Differently from autoregressive LLMs most image transformers are used for one shot inference. Meaning, given one input image the entire output is generated with one single inference. Sometimes instead, the input image needs to be partitioned and batched into a number of prompts that are then fed to the model for inference. Given this peculiarity, the obvious choice was to integrate vision transformers as pooling models, exploiting the multimodal input capabilities of vLLM. Images are pre-processed into tensors and fed to vLLM for inference on the model. An `identity` pooler returning the raw model hidden states is used, and the model output can be post-processed into images.

The geospatial vision transformers we are targeting are accessible via the [Terratorch](https://github.com/IBM/terratorch) framework, a wrapper around pytorch targeting geospatial applications. Along the same lines of what was done for Transformer models, we have created a new model implementation backend for serving Terratorch models in vLLM as pooling models.

One example model class that can be served with Terratorch is [Prithvi for Earth observation](ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11).


## Serving the Prithvi Geospatial model for flood detection on vLLM

### Start a vLLM serving instance
Install vLLM in your python environment. At the time of writing this article the changes required for replicating this example are not yet part of a release (current latest is v0.10.1.1) and we advise users to install the [latest code](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#install-the-latest-code_1).

Download and install the IOProcessor plugin for flood detection with Prithvi

```bash
git clone git@github.com:christian-pinto/prithvi_io_processor_plugin.git
cd prithvi_io_processor_plugin
pip install .
```

This will install two plugins: `prithvi_to_tiff_india` and `prithvi_to_tiff_valencia`.

Start a vLLM serving instance that loads the `prithvi_to_tiff_valencia` plugin.

```bash
vllm serve \
    --model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
    --model-impl terratorch \
    --task embed --trust-remote-code \
    --skip-tokenizer-init --enforce-eager \
    --io-processor-plugin prithvi_to_tiff_valencia
```

Look for the below lines in the vLLM log to confirm the plugins are properly installed in the python environment

```bash
[__init__.py:36] Available plugins for group vllm.io_processor_plugins:
[__init__.py:38] - prithvi_to_tiff_india -> prithvi_io_processor:register_prithvi_india
[__init__.py:38] - prithvi_to_tiff_valencia -> prithvi_io_processor:register_prithvi_valencia
```

Once the serving instance is fully up and running it is ready to serve requests with the selected plugin. The below log entries confirm you vLLM instance is up and running and that it is listening on port `8000`

```bash
(APIServer pid=409128) INFO 09-04 13:12:10 [api_server.py:1969] Starting vLLM API server 0 on http://0.0.0.0:8000
...
...
(APIServer pid=409128) INFO:     Started server process [409128]
(APIServer pid=409128) INFO:     Waiting for application startup.
(APIServer pid=409128) INFO:     Application startup complete.
```

## Send requests to the model
The below script sends a request to the vLLM pooling endpoint where the `priority`, `model` and `softmax` arguments are pre-defined, while the `data` field is defined by the user and depends on the plugin in use. In this case we send the input image to vLLM as a URL and we request the response image to be a base64 encoded image. The script decodes the image and writes it to disk as a tiff(geotiff) file

```python
import base64
import os
import requests

def main():
  image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"
  server_endpoint = "http://localhost:8000/pooling"

  request_payload_url = {
      "data": {
          "data": image_url,
          "data_format": "url",
          "image_format": "tiff",
          "out_data_format": "b64_json",
      },
      "priority": 0,
      "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
      "softmax": False,
  }

  ret = requests.post(server_endpoint, json=request_payload_url)

  if ret.status_code == 200:
    response = ret.json()

    decoded_image = base64.b64decode(response["data"]["data"])

    out_path = os.path.join(os.getcwd(), "online_prediction.tiff")

    with open(out_path, "wb") as f:
        f.write(decoded_image)
  else:
    print(f"Response status_code: {ret.status_code}")
    print(f"Response reason:{ret.reason}")


if __name__ == "__main__":
    main()
```

Below is an example of input and output you should obtain showing flooded areas during the 2024 flood in Valencia, Spain.

<p align="center">
<picture>
<img src="/assets/figures/llama31/perf_llama3.png" width="50%">
</picture>

## What’s Next
We’re excited to see how the community uses IOProcessor to push the boundaries of what’s possible with vLLM. Whether you're building a vision-language system, a structured reasoning agent, or a hybrid model pipeline, IOProcessor gives you the tools to serve it—all in one place.

To get started, check out the IOProcessor documentation and explore the examples. Contributions, feedback, and ideas are always welcome!

