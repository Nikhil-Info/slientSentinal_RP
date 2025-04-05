import time
import numpy as np

# Simulate detection latency and throughput for different batch sizes
batch_sizes = [32, 64, 128, 256, 512]
latency = []
throughput = []

for batch_size in batch_sizes:
    start_time = time.time()
    
    # Assuming model.predict takes the batch_size as input
    model.predict(test_data[:batch_size])
    
    end_time = time.time()
    detection_time = end_time - start_time  # in seconds
    
    latency.append(detection_time)
    throughput.append(batch_size / detection_time)  # throughput (samples/sec)

# Plot latency vs throughput
plt.plot(latency, throughput, marker='o')
plt.title('Detection Latency vs. Throughput')
plt.xlabel('Latency (s)')
plt.ylabel('Throughput (samples/sec)')
plt.show()
