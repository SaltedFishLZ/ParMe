


NUM_TEMP=1
THETA=0.0

RHO_STAR=92426025394.58632
PHI_STAR=4319.399584121471


mkdir -p experiments

for NUM_TEMP in 1
do
    for THETA in 0.001 0.999
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