# P4 (S10): the ComfyUI measurement of M6 again, the repair now coming from the official DLC entail-dlc-comfyui.
# Runs issue_track/comfyui_field_test/vpred_harm.py in both orders (noob_native_node, noob_node_native) with entail
# on, once with the DLC attached (VPRED_TAG=_dlc) and once with ENTAIL_DLC=off (_nodlc). Nothing is installed into
# ComfyUI's environment: the product-branch entail, its start-up hook and the DLC come in through PYTHONPATH.
#   powershell -File testbed/p4_comfy.ps1 <DLC site folder>
param([string]$DlcSite)
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Entail = Join-Path $Repo "entail"
$Hook = Join-Path $Entail "entail\adapters\autoinstall"
$Comfy = "E:\ComfyUI\ComfyUI-new"
$Py = Join-Path $Comfy ".venv\Scripts\python.exe"
$Harm = Join-Path $Repo "issue_track\comfyui_field_test\vpred_harm.py"
$env:COMFY_ROOT = $Comfy
$env:PYTHONPATH = "$Entail;$DlcSite;$Hook"
foreach ($run in @(@{tag = "_dlc"; extra = "{}"}, @{tag = "_nodlc"; extra = '{"ENTAIL_DLC": "off"}'})) {
    foreach ($mode in @("noob_native_node", "noob_node_native")) {
        $env:VPRED_TAG = $run.tag
        $env:COMFY_EXTRA_ENV = $run.extra
        Write-Output "== $mode on $($run.tag)"
        & $Py $Harm $mode on | Select-Object -Last 12
    }
}
