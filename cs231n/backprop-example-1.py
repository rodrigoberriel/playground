import tensorflow as tf

# define the function f
x = tf.Variable([-2.0], name='x')
y = tf.Variable([+5.0], name='y')
z = tf.Variable([-4.0], name='z')
q = tf.add(x, y, name='q')
f = tf.multiply(q, z, name='f')

# define the optmizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads = opt.compute_gradients(f)

with tf.Session() as sess:
    # init vars and save the graph
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter('/tmp/tf/backprop-example-1', sess.graph)

    # run
    output = sess.run({
        'output': f,
        'grads': grads,
    })

    # print grads
    print('Result (f): {:+.0f}'.format(output['output'][0]))
    print('Gradients:')
    for i in range(len(output['grads'])):
        print('- variable: {}\tgrad: {:+.0f}\tvalue: {:+.0f}'.format(
            grads[i][1].name.split(':')[0], output['grads'][i][0][0], output['grads'][i][1][0])
        )
