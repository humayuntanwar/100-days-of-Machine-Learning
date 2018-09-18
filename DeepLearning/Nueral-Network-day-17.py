#import tensorflow
import tensorflow as tf

# 2 constants
x1 = tf.constant(5)
x2 = tf.constant(6)

#multiply
result = tf.multiply(x1,x2)

#test print
print(result)
#session, open session
#sess = tf.Session()
#print(sess.run(result))

# with tf.Session() as sess:
#     print(sess.run(result))


with tf.Session()as sess:
    output = sess.run(result)
    print(output)

print(output)
