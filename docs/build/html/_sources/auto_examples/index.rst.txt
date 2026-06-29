:orphan:

Examples
========

Here are some examples of how to use MFML-QC.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial covers the basics of data handling, visualization, and structural representation generation in MFML-QC. We will use an inbuilt dataset for the benzene molecule.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_01_load_and_represent_thumb.png
    :alt:

  :doc:`/auto_examples/01_load_and_represent`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A Multi-Fidelity Dataset: MD Benzene</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will build our first kernel based machine learning model with the in-built Kernel Ridge Regressor module. We will focus on a single fidelity KRR model that predicts the excitation energies of the benzene molecule from the dataset provided in this package.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_02_krr_learning_curve_thumb.png
    :alt:

  :doc:`/auto_examples/02_krr_learning_curve`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kernel Ridge Regression</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the MFML-QC package to load the built-in Benzene trajectory dataset, manually extract a multi-fidelity subset using a top-down approach (that is start with the highest fidelity then move down the fidelity hierarchy), and train an MFML model to predict high-fidelity excitation energies.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_03_mfml_basics_thumb.png
    :alt:

  :doc:`/auto_examples/03_mfml_basics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Basics of MFML</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the &quot;duck typing&quot; flexibility of the ModelMFML orchestrator. DUck typing is summarized well with &#x27;If it walks like a duck, swims like a duck, and wualks like a duck, it must be a duck.&#x27; Here, we use it to mean that we can use any ML architecture that has certain attributes. While MFML-QC provides an ultra-fast, built-in Kernel Ridge Regressor (KRR), you are not forced into using it.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_04_duck_typing_mfml_thumb.png
    :alt:

  :doc:`/auto_examples/04_duck_typing_mfml`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Duck Typing for MFML</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="As we saw in on of the previous examples, learning curves are a vital metric for evaluating kernel-based machine learning methods. They depict the relationship between the model&#x27;s prediction error and the amount of training data provided.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_05_mfml_learning_curve_thumb.png
    :alt:

  :doc:`/auto_examples/05_mfml_learning_curve`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MFML Cross-Validation and Learning Curves</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the use of optimized multifidelity machine learning (o-MFML) with a comparison of learning curves for:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_06_omfml_thumb.png
    :alt:

  :doc:`/auto_examples/06_omfml`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optimized MFML</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to use the OrcaEngine to perform quantum chemistry calculations.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_07_orca_oracle_thumb.png
    :alt:

  :doc:`/auto_examples/07_orca_oracle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Interfacing with QC Oracles (ORCA)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In machine learning for quantum chemistry, molecular geometries are often abundant but labeling them with the properties (e.g., running expensive quantum chemistry calculations for excitation energies) is costly. Active Learning (AL) solves this by iteratively selecting only the most informative geometries from an unlabeled pool to add to the training set.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_08_pool_based_active_learning_thumb.png
    :alt:

  :doc:`/auto_examples/08_pool_based_active_learning`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Active Learning Protocols</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/01_load_and_represent
   /auto_examples/02_krr_learning_curve
   /auto_examples/03_mfml_basics
   /auto_examples/04_duck_typing_mfml
   /auto_examples/05_mfml_learning_curve
   /auto_examples/06_omfml
   /auto_examples/07_orca_oracle
   /auto_examples/08_pool_based_active_learning


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
