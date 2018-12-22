import psutil
import onnxruntime
import numpy

assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
device_name = 'gpu'

sess_options = onnxruntime.SessionOptions()

# Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
# Note that this will increase session creation time so enable it for debugging only.
sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_{}.onnx".format(device_name))

# Please change the value according to best setting in Performance Test Tool result.
sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)

session = onnxruntime.InferenceSession(export_model_path, sess_options)
