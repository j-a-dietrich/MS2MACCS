![header](imgs/logo.jpg)

MS2MACCS predicts MACCS fingerprints (167 bit vector) from tandem mass spectrometry data (MS2). It does so by combining a look-up table for common fragment structure combined with a transformer architecture.

## ✨ Why use MS2MACCS?
➕➖ predicts fingerprints from spectra measured in *positive* **and** *negative* mode <br>
⚡predicts ~40-300 spectra per second (cpu i7 8th generation vs gpu RTX A5000) <br>
🔝 shows comparable results with other state-of-the art tools (we do not achieve as good results as Sirius)<br>
🆓 MS2MACCS is completely open source (do with it whatever you want)<br>

## 🛠 Installation

```
conda create -n ms2maccs python=3.10
conda activate ms2maccs

git clone https://github.com/j-a-dietrich/MS2MACCS.git
cd MS2MACCS
pip install -e .

# if cuda available
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# if no cuda
pip install torch==2.7.1

```

## 🚀 Quickstart
```
from ms2maccs import MS2MACCS

m = MS2MACCS(
    "../models/standard_model.pt", 
    "../fp_bit_maps/fp_bit_map_H_p_mode.pkl", 
    "../fp_bit_maps/fp_bit_map_H_n_mode.pkl", 
    "cpu", 
)

pred_maccs = m.predict("../ms2_data/test_specs_H_p_mode.mgf").to("cpu")

# see demo/demo.ipynb
```

## 📬 Get in Touch
💡 Questions, ideas, or contributions? Open an issue.

## 📚 Citation
coming at some point (potentially)




