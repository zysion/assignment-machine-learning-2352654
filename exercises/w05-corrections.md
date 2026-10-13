I had a misunderstanding regarding the chain rule and backpropagation.

First, chain rule is a fundamental formula in calculus used to compute the derivative of a composite function, where the output of one function serves as the input to another, i.e., $y = g(f(x))$.To find the derivative of $g$ with respect to $x$, we first differentiate $g$ with respect to $f$, then differentiate $f$ with respect to $x$, and multiply the two resulting derivatives:$$\frac{dg}{dx} = \frac{dg}{df} \cdot \frac{df}{dx}$$

Regarding backpropagation in an MLP: these are the steps taken to update the model's weights and biases based on how changes to them affect the value of the loss function.
$$wnew = w - \eta \frac{\partial L}{\partial w}$$

$$bnew = b - \eta \frac{\partial L}{\partial b}$$
The updated weight or bias value is calculated by subtracting the product of the learning rate and the derivative of the loss function with respect to the weight ($w$) from the current value.
If the derivative of the loss function with respect to $w$ is greater than zero, it means that increasing $w$ causes the loss ($L$) to increase; if the derivative is less than zero, increasing $w$ causes $L$ to decrease.
So, how do we find the derivative of $L$ with respect to $w$?
We apply the chain rule. First, the values ​​of $w$ and $b$ serve as inputs to the function $f(w, b) = wx + b$. The result of function $f$ then becomes the input for the activation function ($a$), and the output of the activation function serves as the input for the loss function. Therefore, to find the derivative of $L$ with respect to $w$ or $b$, we must first calculate the derivative of $L$ with respect to $a$, then the derivative of $a$ with respect to $f$, and finally the derivative of $f$ with respect to $w$ or $b$, before multiplying all three values ​​together.

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial b}$$