NUM_TEMP=3
THETA=0.0

RHO_STAR=488
PHI_STAR=818

mkdir -p experiments

for NUM_TEMP in 3
do
    for THETA in 0.8 0.85 0.9 # 0.4 0.5 0.6
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
