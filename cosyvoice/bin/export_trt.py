# TODO has the same logic as export_jit, and completes the onnx export of the estimator in the flow part. #Installation method of tensorrt, and then write down the steps as follows. If it is not installed, then do not execute this script. The user will be prompted to install it first and will not be given a choice.
try:
    import tensorrt
except ImportError:
    print("step1, 下载\n step2. 解压，安装whl，")
# Import the root directory of tensosrt in the installation command using environment variables, such as os.environ['tensorrt_root_dir']/bin/exetrace, and then execute the export command in subprocess in python
# Later I will write the execution command tensorrt_root_dir= in run.sh xxxx python cozyvoice/bin/export_trt.py --model_dir xxx
