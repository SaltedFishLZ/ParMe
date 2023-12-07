


NUM_TEMP=5
THETA=1.0

mkdir -p experiments

for NUM_TEMP in 5
do
    for THETA in 1.0
    do
        python run_mpdl.py \
        --t ${NUM_TEMP} \
        --threads 16 \
        --time 30 \
        --max-size 200 \
        --min-size 80 \
        --w0 7 13 12 \
        --theta ${THETA} \
        --output experiments/t_${NUM_TEMP}_theta_${THETA}
    done
done