nvidia-smi \
  --query-gpu=timestamp,index,pstate,clocks.current.sm,clocks.max.sm,\
power.draw,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 100
