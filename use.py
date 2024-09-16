from nn import MLP

n = MLP(2, [3, 1])

xs = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
# XOR results in ys (desired targets)
ys = [0.0, 1.0, 1.0, 0.0] 


# Train the network for more iterations
for k in range(1000):
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum([(ygt - yout)**2 for ygt, yout in zip(ys, ypred)])

  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  # update
  for p in n.parameters():
    p.data += -0.01 * p.grad  # smaller learning rate

  if k % 100 == 0:  # print every 100 iterations
    print(k, loss.data)


print(ypred)
