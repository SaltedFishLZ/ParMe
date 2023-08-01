NUM_TEMP=1
THETA=0.0

mkdir -p experiments

for NUM_TEMP in 2 # 3 # 1 2
do
    for THETA in 0.01 # 0.99
    do
        python run_cpu.py \
        --t ${NUM_TEMP} \
        --threads 16 \
        --time 2 \
        --max-size 100 \
        --min-size 50 \
        --w0 10 10 10 10 \
        --theta ${THETA} \
        --output experiments/t_${NUM_TEMP}_theta_${THETA}
    done
done