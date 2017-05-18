import csv as csv


with open('label.csv', 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for s in range(4):
    	s=s+1
    	for g in range(11):
    		g=g+1
    		for i in range(30):
    			i=i+1
    			writer.writerow({'id': str(s)+str(g)+str(i), 'label': str(g)})
