param(
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([System.IO.Path]::GetPathRoot($repo) -ine "D:\") {
    throw "The project must be on drive D."
}
$work = Join-Path $repo ".submission_work\bp-next"
$env:BP_MODEL_DIR = Join-Path $work "models"
$env:PADDLE_PDX_CACHE_HOME = Join-Path $env:BP_MODEL_DIR "paddlex"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
$env:PADDLE_HOME = Join-Path $work "cache\paddle"
$env:HF_HOME = Join-Path $work "cache\huggingface"
$env:MODELSCOPE_CACHE = Join-Path $work "cache\modelscope"
$env:TORCH_HOME = Join-Path $work "cache\torch"
$env:XDG_CACHE_HOME = Join-Path $work "cache\xdg"
$env:MPLCONFIGDIR = Join-Path $work "cache\matplotlib"
$env:NUMBA_CACHE_DIR = Join-Path $work "cache\numba"
$env:UV_CACHE_DIR = Join-Path $work "uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $work "python"
$env:UV_PYTHON_DOWNLOADS = "never"
$env:PIP_CACHE_DIR = Join-Path $work "cache\pip"
if (!$env:BP_VERIFIED_REFERENCES) {
    $env:BP_VERIFIED_REFERENCES = Join-Path $work "knowledge\verified.json"
}
$env:BP_TEXT_API_ONLY = "1"
$env:TEMP = Join-Path $work "temp"
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Set-Location -LiteralPath $repo
$python = Join-Path $work "venv-cpu\Scripts\python.exe"
$cudaPython = Join-Path $work "venv-cuda\Scripts\python.exe"
$env:BP_EDIT_BACKEND = "trufor"
if (Test-Path -LiteralPath $cudaPython) {
    $cudaTorch = Join-Path $work "venv-cuda\Lib\site-packages\torch\__init__.py"
    $gpuReady = $false
    if ((Test-Path -LiteralPath $cudaTorch) -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        & nvidia-smi --list-gpus 2>$null | Out-Null
        $gpuReady = $LASTEXITCODE -eq 0
    }
    if ($gpuReady) {
        $python = $cudaPython
        $env:BP_EDIT_BACKEND = "iml"
    }
}
if (!(Test-Path -LiteralPath $python)) {
    throw "D drive virtual environment not found: $python. See docs/BEAUTY_REVIEW_RUNTIME.md."
}
if ($Check) {
    & $python -B -c "import sys, torch, torchvision, paddleocr, flask; print('Python:', sys.executable); print('Torch:', torch.__version__, torch.__file__); print('CUDA:', torch.cuda.is_available()); print('PaddleOCR:', paddleocr.__file__); print('Flask:', flask.__file__)"
} else {
    & $python -B "web/app.py"
}
exit $LASTEXITCODE
