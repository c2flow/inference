set -e

# Write conda configuration to ~/.condarc
cat > ~/.condarc <<EOF
channels:
- defaults
show_channel_urls: true
auto_update_conda: false
EOF

conda install pybind11==2.10.4 -c conda-forge -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia
python -m pip install transformers==4.46.2 nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece==0.2.0 accelerate==1.2.1


cd ../../loadgen && python3 -m pip install .
