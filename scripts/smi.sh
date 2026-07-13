nvidia-smi \
  --query-gpu=timestamp,index,clocks.current.sm,clocks.max.sm,power.draw,power.limit,utilization.gpu,utilization.memory \
  --format=csv -lms 100
