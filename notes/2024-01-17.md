
# Topics

- Arrays, Tensors, Broadcasting
- Stochastic Gradient Descent
- Loss Function (for classification), role of mini batches
- Describe Math of NN

#### Important Names in ML

Yann Lecun #person
Yoshua Bengio #person 
Geoffrey Hinton #person 
Jurgen Schmidhuber #person 
Sepp Hochreiter #person 
Paul Werbos #person 


# Creating a Handwritten Digit Classifier

Knowing the human brain, I would start with looking at edge detection. I know that CNNs are the best way to actually do this, so I am somewhat stuck in this approach as a thought.

Something cool about this chapter is displaying the pixel values uses pandas df. Can totally see myself doing something like this for Rippey or down the line.

```Python
im3_t = tensor(im3) # some tensor function from fastai

df = pd.DataFrame(im3_t[4:15,4:22]) # grabbing top of image

df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys') # display!
```

#### First Approach - Pixel Similarity
- An interesting Idea. Creating the "Ideal" version of a particular class by taking the average pixel value! Pretty interesting stuff. Then, comparing them to the ideal version of one or the other, that might be a good idea.

Here's a copy of some pytorch code:

For every pixel position, we want to compute the average over all the images of the intensity of that pixel. To do this we first combine all the images in this list into a single three-dimensional tensor. Trev: (literally think of a stack of images on top of one another). The most common way to describe such a tensor is to call it a _rank-3 tensor_. We often need to stack up individual tensors in a collection into a single tensor. Unsurprisingly, PyTorch comes with a function called `stack` that we can use for this purpose. 

Some operations in PyTorch, such as taking a mean, require us to _cast_ our integer types to float types. Since we'll be needing this later, we'll also cast our stacked tensor to `float` now. Casting in PyTorch is as simple as typing the name of the type you wish to cast to, and treating it as a method.

Generally when images are floats, the pixel values are expected to be between 0 and 1, so we will also divide by 255 here:

```Python
stacked_sevens = torch.stack(seven_tensors).float()/255

stacked_threes = torch.stack(three_tensors).float()/255

stacked_threes.shape

# Output: torch.Size([6131, 28, 28]) - 6131 images of threes
```

**_rank_ is the number of axes or dimensions in a tensor; _shape_ is the size of each axis of a tensor.**

Another way to get a tensors rank:

```python
stacked_threes.ndim
```

Moving Forward:

Finally, we can compute what the ideal 3 looks like. We calculate the mean of all the image tensors by taking the mean along dimension 0 of our stacked, rank-3 tensor. This is the dimension that indexes over all the images.

```Python
mean3 = stacked_threes.mean(0)
show_image(mean3);
```

In other words, for every pixel position, this will compute the average of that pixel over all images. Trev: Because the first dimension in this tensor represents the images, averaging them along that axis means squishing all of the image pixels values into one, and finding the average.

The result will be one value for every pixel position, or a single image. Here it is:

![[Screenshot 2024-01-17 at 6.07.31 PM.png]]

Next, we need to calculate the distance between this image and a typical image of "three"

##### How would you do this?

Trev: I think the best way to do this would probably be to find the diff between each pixel value on a given picture.

Nope!

The way to do this is to find 1 of two different values:
- Absolute Value of Differences - Mean Absolute Difference (L1 Norm)
- The Mean of the SQUARE of differences (positive nums only) + Square Root (which undoes squaring) - Root Mean Squared Error or RMSE (L2 Norm)

In raw python:

```Python
dist_3_abs = (a_3 - mean3).abs().mean # L1

dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt() # L2

dist_3_abs,dist_3_sqr
```

These are Loss Functions!

In PyTorch:

```Python
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
```

## Numpy Arrays and PyTorch Tensors

The API provided by PyTorch and Pytorch are similar

That said, PyTorch:
- Has GPU Support
- Calculating Gradients

Anything fast in Python is typically a wrapper around a compiled objected written and optimized in another language, typically C. Numpy Arrays and PyTorch arrays are _much_ faster than pure Python.

Numpy doesn't have strict typing, while PyTorch does. For example, For Numpy arrays, "Since that can be any type at all, they can even be arrays of arrays, with the innermost arrays potentially being different sizes—this is called a "jagged array.""

![[Screenshot 2024-01-17 at 6.25.58 PM.png]]

Numpy Shines when there are different types. Since it is optimized in C, it can be very fast.

PyTorch, on the other hand, has to be a single type for all values. In addition to GPU support, PyTorch can automatically:
- Calculate Derivatives of these operations, including combinations of operations
	- Would be impossible to do ML in practice without this

Create an Array
```Python
data = [[1,2,3],[4,5,6]]

arr = array(data) # numpy

tns = tensor(data) # pytorch
```

Accessing Row or Column
```Python
tns[1] # row
tns[:,1] # column
tns[1,1:3] # second row, columns 2-3
```

Operators work on Tensors
```Python
tns+1 # adds 1 to all values
```

Tensors also have a type
```Python
tns.type() # easier than type(tns), and more explicit
```

They will also automatically change type as need, like float to int.

### Computing Metrics Using Broadcasting

To be continued!
