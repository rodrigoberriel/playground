import tensorflow as tf

# define the function f
w0 = tf.Variable([+2.00], name='w0')
x0 = tf.Variable([-1.00], name='x0')
w1 = tf.Variable([-3.00], name='w1')
x1 = tf.Variable([-2.00], name='x1')
w2 = tf.Variable([-3.00], name='w2')

dot = tf.add_n([tf.multiply(w0, x0), tf.multiply(w1, x1), w2])
f = tf.div(1.0, tf.add(1.0, tf.exp(-dot)))

# define the optmizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads = opt.compute_gradients(f)

with tf.Session() as sess:
    # init vars and save the graph
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter('/tmp/tf/backprop-example-2', sess.graph)

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
