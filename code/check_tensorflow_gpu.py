"""
More detailed TensorFlow GPU diagnostic script
"""
import sys
import os

print("=" * 70)
print("Detailed TensorFlow GPU Diagnostics")
print("=" * 70)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check TensorFlow import
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print(f"\n✓ TensorFlow imported successfully")
    print(f"  Version: {tf.__version__}")
    print(f"  Location: {tf.__file__}")
except Exception as e:
    print(f"\n✗ Failed to import TensorFlow: {e}")
    sys.exit(1)

# Check for GPU libraries
print("\n" + "-" * 70)
print("Checking for CUDA/cuDNN libraries...")
print("-" * 70)

cuda_lib_paths = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
]

found_libs = []
for lib_path in cuda_lib_paths:
    if os.path.exists(lib_path):
        print(f"✓ Found: {lib_path}")
        cuda_libs = [f for f in os.listdir(lib_path) if 'cuda' in f.lower() or 'cudnn' in f.lower()]
        if cuda_libs:
            print(f"  CUDA libraries: {len(cuda_libs)} found")
            found_libs.extend([os.path.join(lib_path, lib) for lib in cuda_libs[:5]])

if not found_libs:
    print("✗ No CUDA libraries found in standard locations")

# Check LD_LIBRARY_PATH
print("\n" + "-" * 70)
print("Environment variables...")
print("-" * 70)
ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
cuda_home = os.environ.get('CUDA_HOME', 'Not set')
cuda_path = os.environ.get('CUDA_PATH', 'Not set')
print(f"LD_LIBRARY_PATH: {ld_path}")
print(f"CUDA_HOME: {cuda_home}")
print(f"CUDA_PATH: {cuda_path}")

# Check TensorFlow build info
print("\n" + "-" * 70)
print("TensorFlow build information...")
print("-" * 70)
try:
    build_info = tf.sysconfig.get_build_info()
    print(f"CUDA version (TF expects): {build_info.get('cuda_version', 'Unknown')}")
    print(f"cuDNN version (TF expects): {build_info.get('cudnn_version', 'Unknown')}")
except:
    print("Could not retrieve build info (may be TF 1.x)")

# List all devices
print("\n" + "-" * 70)
print("Available devices...")
print("-" * 70)
try:
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    
    cpu_count = sum(1 for d in local_devices if d.device_type == 'CPU')
    gpu_count = sum(1 for d in local_devices if d.device_type == 'GPU')
    
    print(f"CPU devices: {cpu_count}")
    print(f"GPU devices: {gpu_count}")
    
    for device in local_devices:
        print(f"\n  Device: {device.name}")
        print(f"    Type: {device.device_type}")
        if device.device_type == 'GPU':
            print(f"    Memory: {device.memory_limit / 1024**3:.2f} GB")
            print(f"    Physical device desc: {device.physical_device_desc}")
    
    if gpu_count == 0:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS:")
        print("=" * 70)
        print("1. Install tensorflow-gpu (for TF 1.x):")
        print("   conda install tensorflow-gpu")
        print("   OR")
        print("   pip install tensorflow-gpu==1.15.0")
        print("")
        print("2. For TensorFlow 2.x, ensure CUDA/cuDNN are compatible:")
        print("   Check: https://www.tensorflow.org/install/source#gpu")
        print("")
        print("3. On AWS SageMaker, ensure:")
        print("   - You're using a GPU instance (ml.p3.xlarge, ml.g4dn.xlarge, etc.)")
        print("   - CUDA libraries are available")
        print("   - Environment variables are set correctly")
        
except Exception as e:
    print(f"Error listing devices: {e}")
    import traceback
    traceback.print_exc()

# Test GPU computation
print("\n" + "-" * 70)
print("Testing GPU computation...")
print("-" * 70)
try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    
    with tf.Session(config=config) as sess:
        # Try to create a simple operation on GPU
        with tf.device('/gpu:0'):
            try:
                a = tf.constant([1.0, 2.0, 3.0], name='test_a')
                b = tf.constant([4.0, 5.0, 6.0], name='test_b')
                c = a + b
                result = sess.run(c)
                print(f"✓ GPU computation test successful!")
                print(f"  Result: {result}")
            except Exception as e:
                print(f"✗ GPU computation failed: {e}")
                print("  Trying CPU fallback...")
                with tf.device('/cpu:0'):
                    a = tf.constant([1.0, 2.0, 3.0], name='test_a')
                    b = tf.constant([4.0, 5.0, 6.0], name='test_b')
                    c = a + b
                    result = sess.run(c)
                    print(f"✓ CPU computation successful: {result}")
except Exception as e:
    print(f"✗ Session creation failed: {e}")

print("\n" + "=" * 70)

