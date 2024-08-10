import requests
url = 'https://raw.githubusercontent.com/ppantele/DeceptionECG/main/DeceptionECG_Functions.py'
r = requests.get(url)
exec(r.content)
print('DeceptionECG tools loaded. Use: \n  1. DeceptionECG_GenPurp(n), n: Number of samples \n      --> To generate new ECG samples. [Output: n*12*1000 nd.array]\n  2. DeceptionECG_DiseaseSpec(input, disease), input: Input signal {n*12*1000 nd.array}, disease: Disease type {"AFIB" or "AMI" or "WPW"} \n      --> To generate new ECG samples. [Output: n*12*1000 nd.array]\n  3. DeceptionECG_SignalPlot(input, signal_n), input: Input signals {n*12*1000 nd.array}, signal_n: Specific signal to plot \n      --> To plot an ECG sample. [Output: Within IDE figure]')
