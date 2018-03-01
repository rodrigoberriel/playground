import tensorflow as tf

# define the function f
x = tf.Variable([+3.00], name='x')
y = tf.Variable([-4.00], name='y')
z = tf.Variable([+2.00], name='z')
w = tf.Variable([-1.00], name='w')

f = tf.multiply(tf.add(tf.multiply(x, y), tf.maximum(z, w)), 2.0)

# define the optmizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads = opt.compute_gradients(f)

with tf.Session() as sess:
    # init vars and save the graph
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter('/tmp/tf/backprop-example-3', sess.graph)

    # run
    output = sess.run({
        'output': f,
        'grads': grads,
    })

    # print grads
    print('Result (f): {:+.2f}'.format(output['output'][0]))
    print('Gradients:')
    for i in range(len(output['grads'])):
        print('- variable: {}\tgrad: {:+.2f}\tvalue: {:+.2f}'.format(
            grads[i][1].name.split(':')[0], output['grads'][i][0][0], output['grads'][i][1][0])
        )
