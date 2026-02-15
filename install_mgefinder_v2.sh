#!/bin/bash
# MGEfinder installation script using micromamba (v2 — snakemake-free)
# Copy-paste friendly

set -e

ENV_NAME="mgefinder_env"

eval "$(micromamba shell hook --shell=bash)"

# -----------------------------------------------------------
# 1. Create environment — NO snakemake (it causes solver hell)
# -----------------------------------------------------------
echo ">>> Creating environment with external tools..."
micromamba create -n mgefinder_env \
    -c conda-forge -c bioconda \
    python=3.9 \
    samtools=1.19 \
    bowtie2=2.5.3 \
    bwa=0.7.18 \
    cd-hit=4.8.1 \
    emboss=6.6.0 \
    -y


micromamba activate ${ENV_NAME}

# -----------------------------------------------------------
# 2. Try pysam via pip; fall back to micromamba if it fails
# -----------------------------------------------------------
echo ">>> Attempting pip install of pysam..."
if pip install 'pysam>=0.15.3' 2>/dev/null; then
    echo ">>> pysam installed via pip — great!"
else
    echo ">>> pip pysam failed — installing via micromamba..."
    micromamba install -n ${ENV_NAME} -c conda-forge -c bioconda pysam -y
fi

# -----------------------------------------------------------
# 3. Install Python dependencies
# -----------------------------------------------------------
echo ">>> Installing Python dependencies..."
pip install \
    'click>=7.0,<9.0' \
    'pandas>=0.25,<2.1' \
    'biopython>=1.75,<1.84' \
    'editdistance' \
    'scipy>=1.4,<1.12' \
    'networkx>=2.4,<3.3' \
    'tqdm>=4.40'

# -----------------------------------------------------------
# 4. Install mgefinder (--no-deps)
# -----------------------------------------------------------
echo ">>> Installing mgefinder..."
pip install --no-deps mgefinder==1.0.6

# -----------------------------------------------------------
# 5. Install snakemake SEPARATELY via pip (avoids conda hell)
#    Only needed if you plan to use 'mgefinder workflow'
# -----------------------------------------------------------
echo ">>> Installing snakemake via pip..."
pip install 'snakemake>=7.0,<8.0'

# -----------------------------------------------------------
# 6. Validate
# -----------------------------------------------------------
echo ""
echo "========================================="
echo "  VALIDATION"
echo "========================================="

echo "--- Python & mgefinder ---"
python -c "import mgefinder; print(f'mgefinder {mgefinder.__version__}')" 2>/dev/null || echo "FAIL: mgefinder import"

echo ""
echo "--- External tools ---"
for tool in samtools bowtie2 bwa cd-hit-est needle snakemake; do
    if command -v \$tool &>/dev/null; then
        echo "OK:   \$tool"
    else
        echo "FAIL: \$tool not found"
    fi
done

echo ""
echo "--- Python packages ---"
python -c "
import click, pandas, Bio, pysam, editdistance, scipy, networkx, tqdm
print(f'click       {click.__version__}')
print(f'pandas      {pandas.__version__}')
print(f'biopython   {Bio.__version__}')
print(f'pysam       {pysam.__version__}')
print(f'scipy       {scipy.__version__}')
print(f'networkx    {networkx.__version__}')
print(f'tqdm        {tqdm.__version__}')
"

echo ""
echo "========================================="
echo "  DONE — activate with:"
echo "  micromamba activate ${ENV_NAME}"
echo "========================================="
