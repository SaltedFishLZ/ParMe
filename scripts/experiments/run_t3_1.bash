


NUM_TEMP=3
THETA=0.0

RHO_STAR=92426049882.7166
PHI_STAR=2373.118625418352


mkdir -p experiments

for NUM_TEMP in 3
do
    for THETA in 0.4 0.45 0.5 0.55 0.6
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