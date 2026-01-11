"""
Utility script to check GPU availability for TensorFlow.
Run this to verify that TensorFlow can detect and use your GPU.
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("=" * 60)
print("TensorFlow GPU Check")
print("=" * 60)

# Check GPU availability
try:
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    
    print("\nAvailable devices:")
    for device in local_device_protos:
        print(f"  - {device.name} ({device.device_type})")
        if device.device_type == 'GPU':
            print(f"    Memory: {device.memory_limit / 1024**3:.2f} GB")
    
    gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
    
    if gpu_devices:
        print(f"\n✓ Found {len(gpu_devices)} GPU device(s):")
        for gpu in gpu_devices:
            print(f"  - {gpu}")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        
        with tf.Session(config=config) as sess:
            with tf.device('/gpu:0'):
                a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
                b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
                c = tf.matmul(a, b)
                result = sess.run(c)
                print(f"✓ GPU computation successful!")
                print(f"  Result shape: {result.shape}")
                print(f"  Result:\n{result}")
    else:
        print("\n✗ No GPU devices found!")
        print("  TensorFlow will use CPU for computation.")
        print("\nPossible issues:")
        print("  1. CUDA/cuDNN not installed or incompatible version")
        print("  2. GPU drivers not installed")
        print("  3. TensorFlow not compiled with GPU support")
        print("\nTo check CUDA version:")
        print("  nvidia-smi  (shows driver version)")
        print("  nvcc --version  (shows CUDA toolkit version)")
        
except Exception as e:
    print(f"\n✗ Error checking GPU: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

