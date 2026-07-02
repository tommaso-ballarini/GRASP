# Avvia FluxText server
CUDA_VISIBLE_DEVICES=2 python flux_text_server.py --port 8767 &
FLUX_TEXT_PID=$!

# Aspetta che sia pronto (carica 58GB, può richiedere minuti)
for i in $(seq 1 90); do
    curl -s "http://127.0.0.1:8767/health" > /dev/null 2>&1 && break
    sleep 10
done

# Lancia text_fix
CUDA_VISIBLE_DEVICES=0 python flux_loop.py \
    --stage text_fix \
    --database database/database.json \
    --output /leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_40

kill $FLUX_TEXT_PID