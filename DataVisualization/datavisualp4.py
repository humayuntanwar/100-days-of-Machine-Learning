#3D graphs with matplotlib

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import math
# using this style
style.use('fivethirtyeight')

# setting up fig and sub plot
fig = plt.figure()
# let python know we going 3d
ax1 = fig.add_subplot(111, projection='3d')

# random x,y,z data
x = list(range(1,1000))
y = [math.log(i, 2) for i in x]
z = [math.sin(i/10) for i in x]


# creating plot
ax1.plot(x,y,z)
#labels set
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')
#plot show
plt.show()