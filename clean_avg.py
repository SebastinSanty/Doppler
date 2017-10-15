freq = [400,450,520,570,580,620,690,715,430,525,685,1000,867,879,900,917]

data = [[0]*1200 for _ in xrange(16)]
avg_values = [[] for _ in xrange(16)]

while True:
	try:
		a,b,c = map(float,raw_input().split())
		if c >= 11:
			for i in xrange(16):
				if abs(freq[i]-b) <= 2:
					data[i][int(a/4)] = max(data[i][int(a/4)],c)
					break
	except:
		break

# print 'This is the cleaned data.Corresponding to each frequency,the array of amplitudes is printed below it.'
# print 'The first value of the array corresponds to time 0,second one corresponds to time 4 and etc..'
# print 'The value 0 signifies that no data was obtained for that time for this frequency'
# print 

# for i in xrange(16):
# 	print freq[i]
# 	print
# 	print data[i]
# 	print

for i in xrange(16):
	temp = []
	for j in xrange(1200):
		if j !=0 and j%120 == 0:
			if len(temp):
				val = sum(temp)/len(temp)
				avg_values[i].append(str(val)[:6])
			else:
				avg_values[i].append(-1)

			temp = []
		else:
			if data[i][j] != 0:
				temp.append(data[i][j])

print 'This is the averaged data.Corresponding to each frequency,the array of averaged amplitudes is printed below it.'
print 'The first value of array corresponds to average of amplitude for first 8min(480 seconds or 120 values)'
print 'The second value of array is for the next 8 minutes and so on..'
print 'The value -1 is when there is no data obtained in that interval for that frequency'
print 

for i in xrange(16):
	print freq[i]
	print avg_values[i]
	print

