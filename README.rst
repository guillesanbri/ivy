..
    ToDo:
    [ ] Change dark theme image to be white letters
    [ ] Remove links from collapsed examples
    [ ] Review typos

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo.png?raw=true#gh-dark-mode-only
   :width: 100%
.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo.png?raw=true#gh-light-mode-only
   :width: 100%

.. raw:: html

    <br/>
    <div align="center">
    <a href="https://github.com/unifyai/ivy/issues">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/issues/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/network/members">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/forks/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/stargazers">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/pulls">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
    </a>
    <a href="https://pypi.org/project/ivy-core">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-core.svg">
    </a>
    <a href="https://github.com/unifyai/ivy/actions?query=workflow%3Adocs">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/ivy/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://github.com/unifyai/ivy/actions?query=workflow%3Atest-ivy">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/ivy/actions/workflows/test-ivy.yml/badge.svg">
    </a>
    <a href="https://discord.gg/sXyFF8tDtm">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    </div>
    <br clear="all" />

.. raw:: html

    <div style="display: block;" align="center">
    <b><a href="">Website</a></b> | <b><a href="">Docs</a></b> | <b><a href="">Examples</a></b> | <b><a href="">Design</a></b> | <b><a href="">FAQ</a></b><br><br>
    
    <b>All of AI, at your fingertips</b>
    
    </div>
    
    <br>
    
    <div align="center">
        <a href="https://jax.readthedocs.io">
            <img width="10%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img width="1%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img width="10%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img width="1%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img width="10%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img width="1%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img width="10%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
    </div>
    
    <br clear="all" />

------------------------------------------------------

Ivy is both a ML transpiler and a framework, currently supporting JAX, TensorFlow, PyTorch and Numpy.

Ivy unifies all ML frameworks 💥 enabling you not only to **write code that can be used with any of these frameworks as the backend**, 
but also to **convert 🔄 any function written in any of them to code in your preferred framework!**

You can check out `Ivy as a transpiler`_ and `Ivy as a framework`_ to learn more about this, try out Ivy
straight away going through the `Setting up Ivy`_ section, or dive deep into Ivy's `Documentation`_ and `Examples`_!

If you would like to contribute, you can join our growing `Community`_ 🌍, check out our `Contributing`_ guide,
and take a look at the `open tasks`_ if you'd like to dive straight in 🧑‍💻 

**lets-unify.ai together 🦾**


Contents
--------

* `Ivy as a transpiler`_
* `Ivy as a framework`_
* `Setting up Ivy`_
* `Documentation`_
* `Examples`_
* `Contributing`_
* `Community`_
* `Citation`_

Ivy as a transpiler
-------------------

Ivy's transpiler allow you to use code from any other framework (and soon, from any other version of the same framework!) in your own code with just one line of code. To do so, TODO. 

You can find more information about Ivy as a transpiler in the docs.

When should I use Ivy as a transpiler?
######################################

If you want to use building blocks published in other frameworks (neural networks, layers, array computing libraries, training pipelines...), you want to integrate code developed in various frameworks, or maybe straight up move code from one framework to another, the transpiler is definitely the tool 🔧 for the job! As the output of transpilation is native code in the target framework, you can use the converted code just as if it was code originally developed in that framework, appliying framework-specific optimizations or tools, making a whole new level of code available to you.

Ivy as a framework
-------------------

TODO Lorem ipsum.
TODO: Ivy API, Ivy stateful API, Ivy Container, Ivy array, slim down examples below to build just one example
TODO: Review this:

**Framework Agnostic Functions**

In the example below we show how Ivy's concatenation function is compatible with tensors from different frameworks.
This is the same for ALL Ivy functions. They can accept tensors from any framework and return the correct result.

.. code-block:: python

    import jax.numpy as jnp
    import tensorflow as tf
    import numpy as np
    import torch

    import ivy

    jax_concatted   = ivy.concat((jnp.ones((1,)), jnp.ones((1,))), -1)
    tf_concatted    = ivy.concat((tf.ones((1,)), tf.ones((1,))), -1)
    np_concatted    = ivy.concat((np.ones((1,)), np.ones((1,))), -1)
    torch_concatted = ivy.concat((torch.ones((1,)), torch.ones((1,))), -1)

.. code-block:: python

    import ivy

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))

    ivy.set_backend('torch')  # change to any backend!
    model = MyModel()
    optimizer = ivy.Adam(1e-4)
    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.mean((out - target)**2)

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
        print('step {} loss {}'.format(step, ivy.to_numpy(loss).item()))

    print('Finished training!')

This example uses PyTorch as a backend framework,
but the backend can easily be changed to your favorite frameworks, such as TensorFlow, or JAX.

TODO: Mention extensions

You can find more information about Ivy as a framework in the docs.

When should I use Ivy as a framework?
######################################

As Ivy supports multiple backends, writing code in Ivy breaks you free from framework limitations. If you want to publish highly flexible code for everyone to use, independently of the framework they are using, or you plan to develop ML-related tools and want them to be interoperable with not only the already existing frameworks, but also with future frameworks, then Ivy is for you!

Setting up Ivy
--------------

Lorem ipsum

ToDo: test each one of these procedures in various platforms, add links

Installing using pip
####################

The easiest way to set up Ivy is to install it using pip with the following command:

.. code-block:: bash

    pip install ivy-core

..

    Keep in mind that this won't install any of the underlying frameworks (you will need at least one to run Ivy!). We will (very) soon offer support for multiple versions, but for now we have pinned Ivy to specific versions, so you'll need to install one of these:

+------------+-----------------+
| Framework  | Pinned version  |
+============+=================+
| PyTorch    | 1.11.0          |
+------------+-----------------+
| TensorFlow | 2.9.1           |
+------------+-----------------+
| JAX        | 0.3.14          |
+------------+-----------------+
| NumPy      | 1.23.0          |
+------------+-----------------+

Docker
######

If you prefer to use containers, we also have pre-built Docker images with all the supported frameworks and some relevant packages already installed, which you can pull from:

.. code-block:: bash

    docker pull unifyai/ivy:latest

.. code-block:: bash

    ToDo: docker with GPU support should be explained here

Obviously, you can also install Ivy from source if you want to take advantage of the latest changes, but we can't ensure that everything will work as expected :sweat_smile:

.. code-block:: bash

    ToDo: instructions


If you want to set up testing and various frameworks it's probably best to check out the **Contributing - Setting Up** page, where OS-specific and IDE-specific instructions and video tutorials to do so are available!


Access to the Transpiler API
############################

If you only want to use Ivy as a framework you can ignore this entirely, but if you want to try out Ivy's transpiler you'll have to sign up for an API key following the steps below. Fear not! This is entirely free, but keep in mind that we are on a limited time alpha, so expect some rough edges and share with us any bug you encounter! 

.. code-block:: bash

    ToDo: add instructions and screenshots to login, setting up the API key etc.


Using Ivy
#########

You can find quite a lot more examples in the corresponding section below, but using Ivy is as simple as:

.. code-block:: python

    ToDo: short code snippet showing multi backend support

.. code-block:: python

    ToDo: short code snippet showing transpilation


Documentation
-------------

Todo: links

The **Ivy Docs page** holds all the relevant information about Ivy's and it's framework API reference. 

There, you will find the **Design** page, which is an user-focused guide about the architecture and the building blocks of Ivy. Likewise, you can take a look at the **Deep dive**, which is oriented towards potential contributors of the code base and explains the nuances of Ivy in full detail 🔎. 

Another important sections of the docs is **Background**, which contextualises the problem Ivy is trying to solve, and explains both (1) why is important to solve this problem and (2) how we are adhering to existing standards to make this happen. 

Lastly, you can also find there the **Related Work** section, which paints a clear picture of the role Ivy plays in the ML stack, comparing it to other existing solutions in terms of functionalities and level.


Examples
--------

(ToDo)

You can use Ivy to gain access to every Machine Learning or Deep Learning project out there, independently on the framework you are using!

.. code-block:: python

    import ivy

    ivy.set_backend('torch')  # change to any backend!
    model = MyModel()
    optimizer = ivy.Adam(1e-4)
    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])

There are a lot more examples in out examples page but feel free to check out some more framework-specific examples here :arrow_down:

.. raw:: html

   <details>
   <summary><h3>I'm using PyTorch&ensp;<img style="height: 1.2em; vertical-align:-20%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png"></h3></summary>
      <blockquote>If you are a PyTorch user, you can use Ivy to execute code from any other framework!
         <details>
            <summary><h4>From TensorFlow</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>

.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
        <details>
            <summary><h4>From Jax</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
        <details>
            <summary><h4>From NumPy</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
     </blockquote>
   </details>

   <details>
   <summary><h3>I'm using TensorFlow&ensp;<img style="height: 1.2em; vertical-align:-20%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png"></h3></summary>
      <blockquote>If you are a PyTorch user, you can use Ivy to execute code from any other framework!
         <details>
            <summary><h4>From PyTorch</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
        <details>
            <summary><h4>From Jax</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
        <details>
            <summary><h4>From NumPy</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
     </blockquote>
   </details>
   
      <details>
   <summary><h3>I'm using Jax&ensp;<img style="height: 1.2em; vertical-align:-20%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/jax_logo.png"></h3></summary>
      <blockquote>If you are a PyTorch user, you can use Ivy to execute code from any other framework!
         <details>
            <summary><h4>From PyTorch</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
        <details>
            <summary><h4>From TensorFlow</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
        <details>
            <summary><h4>From NumPy</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
     </blockquote>
   </details>
   
      <details>
   <summary><h3>I'm using NumPy&ensp;<img style="height: 1.2em; vertical-align:-20%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/numpy_logo.png"></h3></summary>
      <blockquote>If you are a PyTorch user, you can use Ivy to execute code from any other framework!
         <details>
            <summary><h4>From PyTorch</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
        <details>
            <summary><h4>From TensorFlow</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
        <details>
            <summary><h4>From Jax</h4></summary>
            <blockquote>
               <details>
                  <summary>Any function</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any model</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
               <details>
                  <summary>Any library</summary>
                  
.. code-block:: python

    import ivy

    # ToDo: Write code

.. raw:: html

               </details>
            </blockquote>
        </details>
        
     </blockquote>
   </details>

   <h3>I'm using Ivy&ensp;<img style="height: 1.75em; vertical-align:-40%" src="https://saas.lets-unify.ai/static/ivy.svg"></h3>
   
Or you can use Ivy as a framework, breaking yourself (and your code) free from deciding which community to support, allowing anyone to run your code in their framework of choice!

.. code-block:: python

    import ivy

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))

    ivy.set_backend('torch')  # change to any backend!
    model = MyModel()
    optimizer = ivy.Adam(1e-4)
    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.mean((out - target)**2)

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
        print('step {} loss {}'.format(step, ivy.to_numpy(loss).item()))

    print('Finished training!')

.. _docs: https://lets-unify.ai/ivy
.. _Colabs: https://drive.google.com/drive/folders/16Oeu25GrQsEJh8w2B0kSrD93w4cWjJAM?usp=sharing
.. _`contributor guide`: https://lets-unify.ai/ivy/contributing.html
.. _`open tasks`: https://lets-unify.ai/ivy/contributing/open_tasks.html

Contributing
------------

We believe that everyone can contribute and make a difference. Whether it's writing code 💻, fixing bugs 🐛, 
or simply sharing feedback 💬, your contributions are definitely welcome and appreciated 🙌 

Check out all of our open tasks, and find out more info in our `Contributing <https://lets-unify.ai/ivy/contributing.html>`_ guide in the docs!

Join our amazing community as a code contributor, and help accelerate our journey to unify all ML frameworks!

.. raw:: html

   <a href="https://github.com/unifyai/ivy/graphs/contributors">
     <img src="https://contrib.rocks/image?repo=unifyai/ivy&anon=0&columns=20&max=100" />
   </a>

Community
------------

ToDo: Add links to discord and twitter

In order to achieve the ambitious goal of unifying AI we definitely need as many hands as possible on it! Whether you are a seasoned developer or just starting out, you'll find a place here! Join the Ivy community in our Discord 👾 server, which is the perfect place to ask questions, share ideas, and get help from both fellow developers and the Ivy Team directly!

Also! Feel free to follow us in Twitter 🐦 as well, we use it to share updates, sneak peeks, and all sorts of relevant news, certainly a great way to stay in the loop 😄

Can't wait to see you there!


Citation
--------

If you use Ivy for your work, please don't forget to give proper credit by including the accompanying paper 📄 in your references. 
It's a small way to show appreciation and help to continue to support this and other open source projects 🙌

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


Background ToDo: Maybe remove this for now?
----------

| (a) `ML Explosion <https://lets-unify.ai/ivy/background/ml_explosion.html>`_
| A huge number of ML tools have exploded onto the scene!
|
| (b) `Why Unify? <https://lets-unify.ai/ivy/background/why_unify.html>`_
| Why should we try to unify them?
|
| (c) `Standardization <https://lets-unify.ai/ivy/background/standardization.html>`_
| We’re collaborating with The `Consortium for Python Data API Standards <https://data-apis.org>`_

Design ToDo: Maybe remove this for now?
------

| Ivy can fulfill two distinct purposes:
|
| 1. Serve as a transpiler between frameworks 🚧
| 2. Serve as a new ML framework with multi-framework support ✅
|
| The Ivy codebase can then be split into three categories, and can be further split into 8 distinct submodules, each of which falls into one of these three categories as follows:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

| (a) `Building Blocks <https://lets-unify.ai/ivy/design/building_blocks.html>`_
| Backend functional APIs ✅
| Ivy functional API ✅
| Backend Handler ✅
| Ivy Compiler 🚧
|
| (b) `Ivy as a Transpiler <https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html>`_
| Front-end functional APIs 🚧
|
| (c) `Ivy as a Framework <https://lets-unify.ai/ivy/design/ivy_as_a_framework.html>`_
| Ivy stateful API ✅
| Ivy Container ✅
| Ivy Array 🚧
