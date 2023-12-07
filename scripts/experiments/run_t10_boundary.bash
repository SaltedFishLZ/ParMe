


NUM_TEMP=10
THETA=0.0

RHO_STAR=92426069021.92918
PHI_STAR=2254.3294858019804


mkdir -p experiments

for NUM_TEMP in 10
do
    for THETA in 0.001 # 0.999
    do
        python run_mpdl.py \
        --t ${NUM_TEMP} \
        --threads 16 \
        --time 30 \
        --max-size 200 \
        --min-size 80 \
        --w0 7 13 12 \
        --rho-star ${RHO_STAR} \
        --phi-star ${PHI_STAR} \
        --theta ${THETA} \
        --output experiments/t_${NUM_TEMP}_theta_${THETA}
    done
done