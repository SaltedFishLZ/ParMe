


NUM_TEMP=2
THETA=0.0

RHO_STAR=92426068136.93102
PHI_STAR=2625.940550103962


mkdir -p experiments

for NUM_TEMP in 2
do
    for THETA in 0.1 0.2 0.3 0.7 0.8 0.9
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