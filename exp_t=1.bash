NUM_TEMP=1
THETA=0.0

RHO_STAR=488
PHI_STAR=949

mkdir -p experiments

for NUM_TEMP in 1
do
    for THETA in 0.9 # 0.2 0.4 0.6 0.8
    do
        python run_cpu.py \
        --t ${NUM_TEMP} \
        --threads 16 \
        --time 2 \
        --max-size 100 \
        --min-size 50 \
        --w0 10 10 10 10 \
        --theta ${THETA} \
        --rho-star ${RHO_STAR} \
        --phi-star ${PHI_STAR} \
        --output experiments/t_${NUM_TEMP}_theta_${THETA}
    done
done