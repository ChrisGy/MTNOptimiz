import pandas as pd
import math

definitions = [None,10/971.82,12/(2.5*1024),None,5/1024,3/471.7,10/616.32,5/292.4]

csv = pd.read_csv('data.csv')

for i,v in enumerate(csv['k']):
	print("***",i)
	print(csv['k'][i])
	csv['k'][i] = definitions[int(csv['Pref (primary)'][i])]

print(csv)
csv.to_csv('data21.csv')
# print(csv['k'])

146.2+146.2+175.44+116.97+204.68+87.72+292.4
