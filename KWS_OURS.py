#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse

parser = argparse.ArgumentParser(description='KWS_OURS')
parser.add_argument('--ready_for_hls',
                    action='store_true',
                    help='perform script until ready for hls')
parser.add_argument('--skip_ready_for_hls',
                    action='store_true',
                    help='perform script after ready for hls')
parser.add_argument('--cppsim',
                    action='store_true',
                    help='perform script until cppsim')
parser.add_argument('--skip_cppsim',
                    action='store_true',
                    help='perform script without cppsim part')
parser.add_argument('--rtlsim',
                    action='store_true',
                    help='perform script until rtlsim')
parser.add_argument('--skip_rtlsim',
                    action='store_true',
                    help='perform script without rtlsim part')
args = parser.parse_args()
print("args: ", args)

from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron

build_dir = "/workspace/finn/export_finn"

# In[3]:

import onnx
from finn.util.test import get_test_model_trained
import brevitas.onnx as bo
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs

if not args.skip_ready_for_hls:
    '''#kws = get_test_model_trained("KWS", 1, 1)
    #bo.export_finn_onnx(kws, (1, 16, 16000, 1), build_dir + "/ckpt.t7.M5_11111.pth.finn.onnx")
    model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111_0902_2.pth.finn.onnx")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(build_dir + "/ckpt.t7.M5_11111.pth.finn_tidy.onnx")'''
    model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111.pth.tidy.onnx")

# In[4]:

#showInNetron(build_dir + "/ckpt.t7.M5_11111_0926.pth.tidy.onnx")

# In[5]:

from finn.util.pytorch import ToTensor
from finn.transformation.merge_onnx_models import MergeONNXModels
from finn.core.datatype import DataType

if not args.skip_ready_for_hls:
    #model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111_0920.pth.finn.onnx")
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)
    # preprocessing: torchvision's ToTensor divides uint8 inputs by 255
    totensor_pyt = ToTensor()
    chkpt_preproc_name = build_dir + "/ckpt.t7.M5_11111.pth.finn_preproc.onnx"
    bo.export_finn_onnx(totensor_pyt, ishape, chkpt_preproc_name)

    # join preprocessing and core model
    pre_model = ModelWrapper(chkpt_preproc_name)
    model = model.transform(MergeONNXModels(pre_model))
    # add input quantization annotation: UINT8 for all BNN-PYNQ models
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType.UINT8)
    #model.set_tensor_datatype(global_inp_name, DataType.INT16)

# In[6]:

from finn.transformation.insert_topk import InsertTopK
from finn.transformation.infer_datatypes import InferDataTypes
from finn.core.datatype import DataType

if not args.skip_ready_for_hls:
    # postprocessing: insert Top-1 node at the end
    model = model.transform(InsertTopK(k=1))
    chkpt_name = build_dir + "/ckpt.t7.M5_11111.pth.finn_pre_post.onnx"
    # tidy-up again
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(chkpt_name)

#showInNetron(build_dir+"/ckpt.t7.M5_11111.pth.finn_pre_post.onnx")

# In[7]:

from finn.util.visualization import showSrc
from finn.transformation.streamline import Streamline
if not args.skip_ready_for_hls:
    showSrc(Streamline)

# In[8]:

from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
import finn.transformation.streamline.absorb as absorb

if not args.skip_ready_for_hls:
    model = ModelWrapper(build_dir +
                         "/ckpt.t7.M5_11111.pth.finn_pre_post.onnx")
    # move initial Mul (from preproc) past the Reshape
    model = model.transform(MoveScalarLinearPastInvariants())
    chkpt_name = build_dir + "/ckpt.t7.M5_11111.pth.finn_pre_stream.onnx"
    model.save(chkpt_name)
    #no use!!the picture is the same as last one
    #showInNetron(build_dir+"/ckpt.t7.M5_11111.pth.finn_pre_stream.onnx")

# In[9]:

if not args.skip_ready_for_hls:
    # streamline
    #model = model.transform(Streamline())
    model = model.transform(Streamline())
    model.save(build_dir + "/ckpt.t7.M5_11111.pth.finn_streamlined.onnx")
    #showInNetron(build_dir+"/ckpt.t7.M5_11111.pth.finn_streamlined.onnx")

# In[10]:

from finn.transformation.streamline import Streamline
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.general import RemoveUnusedTensors

if not args.skip_ready_for_hls:
    model = ModelWrapper(build_dir +
                         "/ckpt.t7.M5_11111.pth.finn_streamlined.onnx")
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model.save(build_dir +
               "/ckpt.t7.M5_11111.pth.finn_before_second_streamlined.onnx")

# In[11]:

#showInNetron(build_dir+"/ckpt.t7.M5_11111.pth.finn_before_second_streamlined.onnx")

# In[12]:

if not args.skip_ready_for_hls:
    model = model.transform(Streamline())
    model.save(build_dir +
               "/ckpt.t7.M5_11111.pth.finn_after_second_streamlined.onnx")

# In[13]:

#showInNetron(build_dir+"/ckpt.t7.M5_11111.pth.finn_after_second_streamlined.onnx")

# In[14]:

if not args.skip_ready_for_hls:
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    model.save(build_dir +
               "/ckpt.t7.M5_11111.pth.finn_ready_for_hls_conversion.onnx")

if args.ready_for_hls:
    exit()

# In[15]:

#showInNetron(build_dir+"/ckpt.t7.M5_11111.pth.finn_ready_for_hls_conversion.onnx")

# In[16]:

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition, )

if not args.skip_cppsim:
    """
    from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
    from finn.custom_op.registry import getCustomOp
    from finn.transformation.infer_data_layouts import InferDataLayouts
    from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
    from finn.core.modelwrapper import ModelWrapper
    from finn.custom_op.registry import getCustomOp
    from finn.util.basic import make_build_dir
    from finn.util.visualization import showInNetron
    import onnx
    from finn.util.test import get_test_model_trained
    import brevitas.onnx as bo
    from finn.core.modelwrapper import ModelWrapper
    from finn.transformation.infer_shapes import InferShapes
    from finn.transformation.fold_constants import FoldConstants
    from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
    from finn.transformation.streamline import Streamline
    from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
    from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
    import finn.transformation.streamline.absorb as absorb
    from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
    from finn.transformation.infer_data_layouts import InferDataLayouts
    from finn.transformation.general import RemoveUnusedTensors
    build_dir = "/workspace/finn"
    """
    # choose the memory mode for the MVTU units, decoupled or const
    mem_mode = "decoupled"

    model = ModelWrapper(
        build_dir + "/ckpt.t7.M5_11111.pth.finn_ready_for_hls_conversion.onnx")
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    # TopK to LabelSelect
    model = model.transform(to_hls.InferLabelSelectLayer())

# In[17]:

if not args.skip_cppsim:
    # input quantization (if any) to standalone thresholding
    model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(to_hls.InferConvInpGen())
    model.save(build_dir + "/ckpt.t7.M5_11111.pth.finn_test.onnx")
    #showInNetron(build_dir+"/ckpt.t7.M5_11111.pth.finn_test.onnx")

# In[18]:

from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

if not args.skip_cppsim:
    model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111.pth.finn_test.onnx")
    model = model.transform(to_hls.InferStreamingMaxPool())
    # get rid of Reshape(-1, 1) operation between hlslib nodes
    model.save(build_dir + "/ckpt.t7.M5_11111.pth.BRemoveCNVtoFCFlatten.onnx")
    model = model.transform(RemoveCNVtoFCFlatten())
    model.save(build_dir + "/ckpt.t7.M5_11111.pth.ARemoveCNVtoFCFlatten.onnx")
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())

# In[19]:

#showInNetron(build_dir + "/ckpt.t7.M5_11111.pth.BRemoveCNVtoFCFlatten.onnx")

# In[20]:

from finn.custom_op.registry import getCustomOp

if not args.skip_cppsim:
    # infer tensor data layouts
    model = model.transform(InferDataLayouts())
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir +
                      "/ckpt.t7.M5_11111.pth.finn_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type(
        "StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    # save the dataflow partition with a different name for easier access
    dataflow_model = ModelWrapper(dataflow_model_filename)
    dataflow_model.save(build_dir +
                        "/ckpt.t7.M5_11111.pth.finn_dataflow_model.onnx")

# In[21]:

#showInNetron(build_dir + "/ckpt.t7.M5_11111.pth.finn_dataflow_model.onnx")

# In[22]:

#showInNetron(build_dir + "/ckpt.t7.M5_11111.pth.finn_dataflow_parent.onnx")

# In[23]:

#showInNetron("/tmp/finn_dev_lab2312/dataflow_partition0_ewnvlgpe/df_model.onnx")

# In[24]:
'''from finn.custom_op.registry import getCustomOp
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node.get_nodeattr("model")
showInNetron(dataflow_model_filename)'''

# In[25]:

from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron
import onnx
from finn.util.test import get_test_model_trained
import brevitas.onnx as bo
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs

if not args.skip_cppsim:
    build_dir = "/workspace/finn/export_finn"
    model = ModelWrapper(build_dir +
                         "/ckpt.t7.M5_11111.pth.finn_dataflow_model.onnx")
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")

# In[26]:

#showInNetron(build_dir + "/ckpt.t7.M5_11111.pth.finn_dataflow_parent.onnx")

# In[27]:

#showInNetron(build_dir + "/ckpt.t7.M5_11111.pth.finn_dataflow_model.onnx")

# In[28]:

# each tuple is (PE, SIMD, in_fifo_depth, ramstyle) for a layer
'''
folding = [
    (1, 1, 128, "auto"),
    (1, 1, 128, "auto"),
    (1, 1, 128, "auto"),
    (1, 1, 128, "auto"),
    (1, 1, 128, "auto"),
    (1, 1, 128, "auto"),
    (1, 1, 128, "auto"),
    (1, 1, 128, "auto"),
    (1, 1, 128, "auto"),
    (1, 1, 3, "auto"),
]
'''
folding = [
    (8, 1, 128, "block"),
    (8, 4, 32, "block"),
    (8, 4, 32, "block"),
    (8, 4, 32, "block"),
    (8, 4, 32, "block"),
    (8, 4, 32, "block"),
    (8, 4, 32, "block"),
    (8, 4, 128, "block"),
    (8, 4, 128, "block"),
    (1, 8, 3, "block"),
]

# In[29]:

URAM_counter = 0

# In[30]:

if not args.skip_cppsim:
    '''for fcl, (pe, simd, ififodepth, ramstyle) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)

        if URAM_counter < 5:
            #fcl_inst.set_nodeattr("runtime_writeable_weights", 1)
            fcl_inst.set_nodeattr("mem_mode", "external")
        else:
            fcl_inst.set_nodeattr("mem_mode", "decoupled")
            fcl_inst.set_nodeattr("ram_style", ramstyle)
        URAM_counter +=1
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififodepth)'''
    #fcl_inst.set_nodeattr("mem_mode", "external")
    #fcl_inst.set_nodeattr("ram_style", ramstyle)

    for fcl, (pe, simd, ififodepth, ramstyle) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        '''if URAM_counter < 7:
            fcl_inst.set_nodeattr("runtime_writeable_weights", 1)
            print(URAM_counter)
        URAM_counter +=1'''

        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififodepth)
        fcl_inst.set_nodeattr("ram_style", ramstyle)

# In[31]:

if not args.skip_cppsim:
    URAM_counter = 0
    # use same SIMD values for the sliding window operators
    #swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator1D")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        print(len(swg_layers))
        simd = folding[i][1]
        if URAM_counter < 1:
            ram_style = "ultra"
        else:
            ram_style = "distributed"
        print(URAM_counter)
        URAM_counter += 1

        swg_inst.set_nodeattr("SIMD", simd)
        swg_inst.set_nodeattr("ram_style", ram_style)

    model = model.transform(GiveUniqueNodeNames())
    model.save(build_dir + "/ckpt.t7.M5_11111.pth.finn_folded.onnx")

#set for cppsim

from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.general import GiveUniqueNodeNames

from finn.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

if not args.skip_cppsim:
    model_for_cppsim = model
    model_for_cppsim = model_for_cppsim.transform(GiveUniqueNodeNames())
    model_for_cppsim = model_for_cppsim.transform(PrepareCppSim())
    model_for_cppsim = model_for_cppsim.transform(CompileCppSim())

    model_for_cppsim.save(build_dir +
                          "/ckpt.t7.M5_11111.pth.finn_for_cppsim.onnx")

    fc0 = model_for_cppsim.graph.node[1]
    fc0w = getCustomOp(fc0)
    code_gen_dir = fc0w.get_nodeattr("code_gen_dir_cppsim")
    print(code_gen_dir)

    model_for_cppsim = model_for_cppsim.transform(SetExecMode("cppsim"))
    model_for_cppsim.save(build_dir +
                          "/ckpt.t7.M5_11111.pth.finn_for_cppsim.onnx")

if args.cppsim:
    exit()

#set for rtlsim

from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

if not args.skip_rtlsim:

    test_fpga_part = "xc7z020clg400-1"
    target_clk_ns = 10

    model_for_rtlsim = ModelWrapper(build_dir +
                                    "/ckpt.t7.M5_11111.pth.finn_folded.onnx")
    model_for_rtlsim = model_for_rtlsim.transform(InsertDWC())
    model_for_rtlsim = model_for_rtlsim.transform(InsertFIFO())
    model_for_rtlsim = model_for_rtlsim.transform(GiveUniqueNodeNames())
    model_for_rtlsim = model_for_rtlsim.transform(
        PrepareIP(test_fpga_part, target_clk_ns))
    model_for_rtlsim = model_for_rtlsim.transform(HLSSynthIP())
    model_for_rtlsim = model_for_rtlsim.transform(
        CreateStitchedIP(test_fpga_part, target_clk_ns))
    model_for_rtlsim = model_for_rtlsim.transform(PrepareRTLSim())
    model_for_rtlsim.set_metadata_prop("exec_mode", "rtlsim")
    model_for_rtlsim.save(build_dir +
                          "/ckpt.t7.M5_11111.pth.finn_for_rtlsim.onnx")

if args.rtlsim:
    exit()

#import finn.core.onnx_exec as oxe
#parent_model = ModelWrapper(build_dir+"/ckpt.t7.M5_11111.pth.finn_dataflow_parent.onnx")
##sdp_node = parent_model.graph.node[1]
#sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
#child_model = build_dir + "/finn_dram_for_cppsim.onnx"
#getCustomOp(sdp_node).set_nodeattr("model", child_model)
#output_dict = oxe.execute_onnx(parent_model, input_dict)
#output_cppsim = output_dict[list(output_dict.keys())[0]]
#
#if np.isclose(output_cppsim, output_golden, atol=1e-3).all():
#    print("Results are the same!")
#else:
#    print("The results are not the same!")

#end of cppsim

# In[32]:

#showInNetron(build_dir + "/ckpt.t7.M5_11111.pth.finn_folded.onnx")

# In[33]:

# print the names of the supported PYNQ boards
from finn.util.basic import pynq_part_map

print(pynq_part_map.keys())

# In[34]:

# change this if you have a different PYNQ board, see list above
pynq_board = "ZCU104"
fpga_part = pynq_part_map[pynq_board]
target_clk_ns = 10

# In[35]:

model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111.pth.finn_folded.onnx")
model = model.transform(ZynqBuild(platform=pynq_board,
                                  period_ns=target_clk_ns))

# In[36]:

model.save(build_dir + "/ckpt.t7.M5_11111.pth.finn_synth.onnx")

# In[37]:

#showInNetron(build_dir + "/ckpt.t7.M5_11111.pth.finn_synth.onnx")

# In[38]:

import os

# set up the following values according to your own environment
# FINN will use ssh to deploy and run the generated accelerator
ip = os.getenv("PYNQ_IP", "192.168.0.3")
username = os.getenv("PYNQ_USERNAME", "xilinx")
password = os.getenv("PYNQ_PASSWORD", "xilinx")
port = os.getenv("PYNQ_PORT", 22)
target_dir = os.getenv("PYNQ_TARGET_DIR", "/home/xilinx/finn_dev_lab2312")
# set up ssh options to only allow publickey authentication
options = "-o PreferredAuthentications=publickey -o PasswordAuthentication=no"

# test access to PYNQ board
#get_ipython().system(' ssh {options} {username}@{ip} -p {port} cat /var/run/motd.dynamic')

# In[39]:

from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ

model = model.transform(
    DeployToPYNQ("192.168.0.3", "22", "xilinx", "xilinx",
                 "/home/xilinx/finn_dev_lab2312"))
model.save(build_dir + "/ckpt.t7.M5_11111.pth.pynq_deploy.onnx")

# In[40]:

target_dir_pynq = target_dir + "/" + model.get_metadata_prop(
    "pynq_deployment_dir").split("/")[-1]
print(target_dir_pynq)
exit()

# In[ ]:

#no need the rest for now
'''
model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111.pth.pynq_deploy.onnx")
sdp_node_middle = getCustomOp(model.graph.node[1])
postsynth_layers = sdp_node_middle.get_nodeattr("model")

#showInNetron(postsynth_layers)


# In[ ]:


model = ModelWrapper(postsynth_layers)
model.model.metadata_props


# In[ ]:


model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111.pth.pynq_deploy.onnx")
model.model.metadata_props


# In[ ]:


get_ipython().system(' ls {model.get_metadata_prop("vivado_pynq_proj")}')


# In[ ]:


model.model.metadata_props


# In[ ]:


target_dir_pynq = '/home/xilinx/finn_dev_lab2312/pynq_deployment_kkl5g6nh'


# In[ ]:


get_ipython().system(" ssh -o PreferredAuthentications=publickey -o PasswordAuthentication=no xilinx@192.168.0.3 -p 22 'ls -l /home/xilinx/finn_dev_lab2312/pynq_deployment_kkl5g6nh'")


# In[ ]:


import pkg_resources as pk
import matplotlib.pyplot as plt
import numpy as np

fn = pk.resource_filename("finn.qnn-data", "KWS/up2.npy")
x = np.load(fn)
x = x.reshape(1, 16000, 1)
plt.imshow(x)


# In[ ]:


from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron
import onnx
from finn.util.test import get_test_model_trained
import brevitas.onnx as bo
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.util.visualization import showInNetron

import numpy as np
from finn.core.onnx_exec import execute_onnx


build_dir = "/workspace/finn"
model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111.pth.pynq_deployonboard.onnx")
iname = model.graph.input[0].name
oname = model.graph.output[0].name
ishape = model.get_tensor_shape(iname)
print(iname)
print("Expected network input shape is " + str(ishape))


# In[ ]:


showInNetron(build_dir + "/ckpt.t7.M5_11111.pth.pynq_deployonboard.onnx")


# In[ ]:


input_dict = {iname: x.reshape(ishape)}
input_dict


# In[ ]:


ret = execute_onnx(model, input_dict, False)


# In[ ]:


ret[oname]


# In[ ]:


get_ipython().system(" ssh -o PreferredAuthentications=publickey -o PasswordAuthentication=no xilinx@192.168.0.3 -p 22 'echo xilinx | sudo -S pip3 install git+https://github.com/fbcotter/dataset_loading.git@0.0.4#egg=dataset_loading'")


# In[ ]:


from finn.core.throughput_test import throughput_test_remote

model = ModelWrapper(build_dir + "/ckpt.t7.M5_11111.pth.pynq_deployonboard.onnx")
res = throughput_test_remote(model, 10000)
print("Network metrics:")
for key in res:
    print(str(key) + ": " + str(res[key]))


# In[ ]:

'''
