# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Deploy Single Shot Multibox Detector(SSD) model
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_
`Leyuan Wang <https://github.com/Laurawly>`_

This article is an introductory tutorial to deploy SSD models with TVM.
We will use GluonCV pre-trained SSD model and convert it to Relay IR
"""
import tvm
from tvm import te

from tvm import relay, autotvm
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils
from tvm.relay.op.annotation import compiler_begin, compiler_end

from tvm.relay import transform

import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime

import timeit

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

import tvm.relay.testing
from tvm.relay.expr_functor import ExprMutator

######################################################################
# Preliminary and Set parameters
# ------------------------------
# .. note::
#
#   We support compiling SSD on both CPUs and GPUs now.
#
#   To get best inference performance on CPU, change
#   target argument according to your device and
#   follow the :ref:`tune_relay_x86` to tune x86 CPU and
#   :ref:`tune_relay_arm` for arm CPU.
#
#   To get best inference performance on Intel graphics,
#   change target argument to :code:`opencl -device=intel_graphics`.
#   But when using Intel graphics on Mac, target needs to
#   be set to `opencl` only for the reason that Intel subgroup
#   extension is not supported on Mac.
#
#   To get best inference performance on CUDA-based GPUs,
#   change the target argument to :code:`cuda`; and for
#   OPENCL-based GPUs, change target argument to
#   :code:`opencl` followed by device argument according
#   to your device.

supported_model = [
    "ssd_512_resnet50_v1_voc",
    "ssd_512_resnet50_v1_coco",
    "ssd_512_resnet101_v2_voc",
    "ssd_512_mobilenet1.0_voc",
    "ssd_512_mobilenet1.0_coco",
    "ssd_300_vgg16_atrous_voc", "ssd_512_vgg16_atrous_coco",
]

model_name = supported_model[5] # download ssd_512_resnet50_v1_voc
dshape = (1, 3, 512, 512)

######################################################################
# Download and pre-process demo image

im_fname = download_testdata(
    "https://github.com/dmlc/web-data/blob/main/" + "gluoncv/detection/street_small.jpg?raw=true",
    "street_small.jpg",
    module="data",
)
#im_fname = plt.imread("/home/niki/biking_resize.jpg") # try to download other image
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

######################################################################
# Convert and compile model for CPU.

block = model_zoo.get_model(model_name, pretrained=True)


def build(target):
    mod, params = relay.frontend.from_mxnet(block, {"data": dshape})
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    return lib

# ######################################################################
# # Create TVM runtime and do inference
# # .. note::
# #
# #   Use target = "cuda -libs" to enable thrust based sort, if you
# #   enabled thrust during cmake by -DUSE_THRUST=ON.

# Annotator

# #############

def run(lib, dev):
    # Build TVM runtime
    m = graph_executor.GraphModule(lib["default"](dev))
    tvm_input = tvm.nd.array(x.asnumpy(), device=dev)
    m.set_input("data", tvm_input)
    # execute
    elapsed_time = timeit.timeit(m.run, number=10)
    print(elapsed_time)
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs


##################################
class GpuConv2dAnnotator(ExprMutator):

    def __init__(self):
        super(GpuConv2dAnnotator, self).__init__()
        self.in_compiler = 0
        self.curr_node = 0
        self.ann_node = 0

    def visit_call(self, call):
        visit = super().visit_call(call)
        self.curr_node += 1
        # if call.op.name == "nn.conv2d":
        #     print(self.curr_node)
        nods = ["nn.conv2d", "nn.bias_add", "nn.relu", "nn.max_pool2d", "add", "transpose",
         "reshape", "nn.softmax", "multiply", "concatenate", "slice_like", "nn.batch_flatten", "strided_slice",
         "greater", "cast", "zeros_like", "ones_like", "where", "nn.l2_normalize", "nn.batch_norm"]
        # if call.op.name == "arange" or call.op.name == "nn.batch_norm":
        #     self.ann_node += 1
        #     return super().visit_call(call)
        if self.curr_node < 140 and call.op.name in nods:
            self.ann_node += 1 
            return relay.annotation.on_device(visit, tvm.cuda())
        print(self.curr_node, self.ann_node, call.op.name)
        # if call.op.name == "nn.conv2d":
        #     print(self.curr_node)

        # if call.op.name == "nn.conv2d":  # Annotate begin at args
        #     self.in_compiler = 1
        #     self.ann_node += 1
        #     return relay.annotation.on_device(visit, tvm.cuda()) 
        # elif (call.op.name == "nn.batch_norm" or call.op.name == "nn.bias_add") and self.in_compiler == 1:
        #     self.in_compiler = 2
        #     self.ann_node += 1 
        #     return relay.annotation.on_device(visit, tvm.cuda())
        # elif (call.op.name == "nn.max_pool" or call.op.name == "nn.relu" or call.op.name == "add" or call.op.name == "nn.batch_norm") and self.in_compiler == 2:
        #     self.in_compiler = 3
        #     self.ann_node += 1
        #     return relay.annotation.on_device(visit, tvm.cuda())
        # elif (call.op.name == "nn.relu" or call.op.name == "nn.max_pool2d" or call.op.name == "add") and self.in_compiler == 3:
        #     self.in_compiler = 0
        #     self.ann_node += 1 
        #     return relay.annotation.on_device(visit, tvm.cuda())
        # if(self.ann_node != self.curr_node):
        #     print(self.in_compiler, self.curr_node, self.ann_node, call.op.name)
        return super().visit_call(call)
##################################

ssd, params = relay.frontend.from_mxnet(block, {"data": dshape})
print(ssd)
sched = GpuConv2dAnnotator()
ssd["main"] = sched.visit(ssd["main"])
ssd = transform.PartitionGraph()(ssd)
ssd = transform.InferType()(ssd)
targets = {"cpu": "llvm", "cuda": "cuda"}
with tvm.transform.PassContext(opt_level=3):
        json, lib, param = relay.build(ssd, target=targets, params=params)
        m = tvm.contrib.graph_executor.create(json, lib, [tvm.cuda(), tvm.cpu()])
        print("Hetero Device:")
        tvm_input = tvm.nd.array(x.asnumpy())
        m.set_input("data", tvm_input)
        elapsed_time = timeit.timeit(m.run, number=10)
        print(elapsed_time)

# for target in ["cuda", "llvm"]:
#     dev = tvm.device(target, 0)
#     if dev.exist:
#         lib = build(target)
#         print("Device:", target)
#         class_IDs, scores, bounding_boxs = run(lib, dev)

#     ######################################################################
#     # Display result

#     ax = utils.viz.plot_bbox(
#         img,
#         bounding_boxs.numpy()[0],
#         scores.numpy()[0],
#         class_IDs.numpy()[0],
#         class_names=block.classes,
#     )
#     plt.show()
