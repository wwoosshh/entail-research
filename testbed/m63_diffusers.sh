#!/bin/bash
# M6.3 on diffusers (testbed/m63_diffusers.py): one process per condition. off/reference: entail not on the path;
# on/observe/strict: ENTAIL=load with the start-up hook from the checkout.
set -u
cd <workspace>
source ~/venvs/gpu/bin/activate
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
HOOK=<workspace>/entail:<workspace>/entail/entail/adapters/autoinstall
REC=/tmp/m63_record.jsonl
plain() { env -u PYTHONPATH -u ENTAIL python testbed/m63_diffusers.py "$@"; }
hooked() { PYTHONPATH=$HOOK ENTAIL=load ENTAIL_RECORD=$REC python testbed/m63_diffusers.py "$@"; }
NOOB=NoobAI-XL-Vpred-v1.0.safetensors
ASTOLFO=entail_test/astolfocarmixVpredxl_acEvo25EP.safetensors
WAI=waiIllustriousSDXL_v160.safetensors
L=/mnt/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Desktop------ai-compiler/1ca95520-f800-4114-8750-ff4f872cd917/scratchpad/m63_loras/nagito_illustrious_v3_peft_names.safetensors
for step in ${*:-i04 m7 s3 lora}; do
  case $step in
    i04) plain gen i04_reference $NOOB reference; plain gen i04_off $NOOB off; hooked gen i04_on $NOOB on
         ENTAIL_POLICY=refuse hooked gen i04_observe $NOOB observe ;;
    m7)  plain gen m7_reference $ASTOLFO reference; plain gen m7_off $ASTOLFO off; hooked gen m7_on $ASTOLFO on ;;
    s3)  plain gen s3_off $WAI off; hooked gen s3_on $WAI on ;;
    lora) plain lora lora_right_off $WAI nagito/nagito_illustrious_v3.safetensors off
          hooked lora lora_right_on $WAI nagito/nagito_illustrious_v3.safetensors on
          plain lora lora_other_off $WAI nagito/nagito_anima_e10.safetensors off
          hooked lora lora_other_on $WAI nagito/nagito_anima_e10.safetensors on
          ENTAIL_ON_BROKEN=stop hooked lora lora_other_strict $WAI nagito/nagito_anima_e10.safetensors strict ;;
    vae) M=<workspace>/testbed/results/m63/manifests
         plain vae vae_reference $WAI reference; plain vae vae_off $WAI off
         ENTAIL_MANIFESTS=$M hooked vae vae_on $WAI on
         ENTAIL_MANIFESTS=$M ENTAIL_POLICY=refuse hooked vae vae_observe $WAI observe ;;
    i01) plain lora i01_off $WAI $L off; hooked lora i01_on $WAI $L on
         ENTAIL_ON_BROKEN=stop hooked lora i01_strict $WAI $L strict ;;
    # after the M6.3 fix of how the adapter reads a LoRA (unet_config): the runs entail took part in, again
    rerun_on) hooked lora lora_right_on $WAI nagito/nagito_illustrious_v3.safetensors on
              hooked lora i01_on $WAI $L on
              ENTAIL_ON_BROKEN=stop hooked lora i01_strict $WAI $L strict ;;
  esac
done
