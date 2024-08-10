!pip -q install deepfake-ecg
import requests
url = 'https://raw.githubusercontent.com/ppantele/DeceptionECG/main/DeceptionECG_Functions.py'
r = requests.get(url)
exec(r.content)
print('DeceptionECG tools loaded. Use: \n  1. DeceptionECG_GenPurp(n), n: Number of samples --> To generate new ECG samples. [Output: n*12*1000 nd.array]\n  2. DeceptionECG_DiseaseSpec(input, disease), input: Input signal {n*12*1000 nd.array}, disease: Disease type {"AFIB" or "AMI" or "WPW"} --> To generate new ECG samples. [Output: n*12*1000 nd.array]')
