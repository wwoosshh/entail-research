source ~/venvs/vllm/bin/activate; export PATH=/usr/local/cuda/bin:$PATH
R=$HOME/ai_compiler/lowlevel/l5/guarantee/results
S=$HOME/ai_compiler/lowlevel/l5/guarantee/scratch
F=$R/freeze_v3.json
echo "post-v3 start $(date)"
# 1. supplementary (after the freeze): the guarantee's graph configuration at gpu_memory_utilization 0.80, off too
cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
git log --oneline -1
BFG_ENGINE_CONFIGS=graphs BFG_ENGINE_MODES=guarantee BFG_ENGINE_GPU_MEM=0.80 BFG_FREEZE=$F \
  python eval/block_fp8_guarantee/engine_smoke.py $R/engine_v3_post_mem080 2>&1 | grep --line-buffered -v "WARNING\|INFO" | cut -c1-300
echo "supplementary done $(date)"
# 2. diagnosis: v2 (ae4fb02, check static under ENTAIL=guarantee) against v3 (2a51daf+, ENTAIL=structure), graphs, 0.55,
#    alternating, two rounds each, in this one session
if [ ! -d ~/entail_ae4fb02 ]; then git clone -q --no-checkout ~/ai_compiler/entail ~/entail_ae4fb02 && (cd ~/entail_ae4fb02 && git checkout -q ae4fb02); fi
(cd ~/entail_ae4fb02 && git log --oneline -1)
for round in a b; do
  cd ~/entail_ae4fb02; export PYTHONPATH=$HOME/entail_ae4fb02
  BFG_ENGINE_CONFIGS=graphs BFG_FREEZE=$R/freeze_v2.json python eval/block_fp8_guarantee/engine_smoke.py $S/engine_diag/v2_$round 2>&1 \
    | grep --line-buffered -v "WARNING\|INFO" | grep --line-buffered "steady_ratio\|exit" | cut -c1-200
  cd ~/ai_compiler/entail; export PYTHONPATH=$HOME/ai_compiler/entail
  BFG_ENGINE_CONFIGS=graphs BFG_ENGINE_MODES=structure BFG_FREEZE=$F python eval/block_fp8_guarantee/engine_smoke.py $S/engine_diag/v3_$round 2>&1 \
    | grep --line-buffered -v "WARNING\|INFO" | grep --line-buffered "steady_ratio\|exit" | cut -c1-200
done
echo "post-v3 done $(date)"
