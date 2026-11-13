"""Zip the code for Colab: colab_code.zip (scripts, kit/, folds.csv). Upload it, with the competition's data zip
(upi-fraud-rings-gradient-wars.zip), to My Drive/upi-fraud-rings/ and run upi_fraud_rings_colab.ipynb."""
import glob, zipfile

files = sorted(glob.glob('*.py') + glob.glob('kit/*.py') + ['folds.csv', 'requirements.txt'])
with zipfile.ZipFile('colab_code.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f)
print(f'colab_code.zip: {len(files)} files')
