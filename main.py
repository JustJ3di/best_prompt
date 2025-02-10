import json
import numpy as np




key_check = ["check_copilot_bandit_semgrep","check_gemini_bandit_semgrep","check_deepseekminmax_bandit-semgrep","check_deepseek_bandit_semgrep","check_gpt4_bandit_semgrep","check_gpt3.5_bandit_semgrep"]


#FUNZIONI DI UTILITà
def factorial(a):
    result = 1
    for k in range(2,n+1):
        result *= k
    return result

def binomial(n, k):
    a, b = (k, n-k) if k < n-k else (n-k, k)
    numerator = 1
    for i in range(b+1, n+1):
        numerator *= i
    return numerator / factorial(a)

def metric_pass(n,k,c):
    num = binomial(n-c,k)
    den = binomial(n,k)
    return 1- num/den

with open("csvjson.json","r") as file:
    data = json.load(file)


'''
indice 0 Basic prompt
indice 3 naive
indice 6 CWE-Specific
indice 9 Comprehensive
indice 12 Persona/Memetic 
'''

#PASS@K per ogni prompt per gli indici vedere tabella di sopra.
result =[]
for i in data:
    print(i["id"])
    c = 0 #correct code
    n = 6 #total code
    k = 1 #
    index = 12
    for key in key_check:
        
        if(i[key][index]!=False):
            c += 1
        
    result.append(metric_pass(n,k,c))
    


print("Pass@k "+str(np.mean(result)))

all_pass = [0.7407407407407407,0.7685185185185186,0.7685185185185186,0.7685185185185186,0.7314814814814814]


result.clear()

#Vulnerabl@K per ogni prompt per gli indici vedere tabella di sopra.The vulnerable@k metric measures the probability that at least one code snippet out of k generated samples is vulnerable.the model is better if the vulnerable@k score is lower.
result =[]
for i in data:
    
    c = 0 #vulneabiliti found
    n = 6 #total code
    k = 1 #
    index = 13
    for key in key_check:
        if((i[key][index-1]!= False) and (i[key][index]!=0 or i[key][index+1] !=0)):
            c += 1
        
    result.append(metric_pass(n,k,c))

print(np.mean(result))

all_vulnerable = [0.5462962962962963,0.4907407407407407,0.5092592592592592,0.5277777777777778,0.4907407407407407]


ordered_data = sorted(data,key=lambda data: data.get('id'))

print(len(ordered_data))


prompt_tecnique = ["Basic","Naive","CWE-Specific","Comprehensive","Persona/Memetic"] 


basis_statistic = {}


for i in ordered_data:
    basis_statistic.update({i["id"]:[0,0,0,0,0]})


def mean(a,b):
    return (a+b)/2

for i in ordered_data:
    tmp = []
    for j in key_check:
        print(i[j])
        if(i[j][1]!=-1 and i[j][2] != -1):
            tmp.append(mean(i[j][1],i[j][2])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_basic = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        print(i[j])
        if(i[j][4]!=-1 and i[j][5] != -1):
            tmp.append(mean(i[j][4],i[j][5])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_naive = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        print(i[j])
        if(i[j][7]!=-1 and i[j][8] != -1):
            tmp.append(mean(i[j][7],i[j][8])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_Comp = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        print(i[j])
        if(i[j][10]!=-1 and i[j][11] != -1):
            tmp.append(mean(i[j][10],i[j][11])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_CWE = np.mean(tmp)
    tmp.clear()

    for j in key_check:
        print(i[j])
        if(i[j][13]!=-1 and i[j][14] != -1):
            tmp.append(mean(i[j][13],i[j][14])) #bisogna inserire un punteggio negativo quando il codice non funziona ?
        else:
            tmp.append(10)
    med_meme = np.mean(tmp)


    for t in tmp:
        basis_statistic.update({i["id"]:[med_basic,med_naive,med_CWE,med_Comp,med_meme]})
  

        

#{id(cwe):[ARRAY CON I RISULTATI]}
        

print(basis_statistic)
with  open("new.json","w") as fp:
    json.dump(basis_statistic,fp=fp)

with open("prompt.json","w") as fp:
    json.dump(ordered_data,fp = fp)