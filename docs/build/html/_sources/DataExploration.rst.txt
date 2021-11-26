Data exploration
================


.. code-block:: python

	df.shape

To see the shape of the dataset : 307511 rows and 122 columns.   

.. code-block:: python
	
	train.info(max_cols = 130)

To see the name of each column, the type and the number of non-null cells::

	pd.set_option("display.max_columns",None)
	train.head(10)


To see the first 10 rows of data, with all the columns. It help to see what can be se values for each feature.

.. code-block:: python

	target = train.TARGET
	count = target.value_counts()
	percentage = target.value_counts(normalize = True)*100
	pd.DataFrame({'counts': count, 'percentage' : percentage})

.. image:: ../images/0.png
    :width: 49 %


To visualize the split of target value = 1 and target value = 0

.. code-block:: python

	plt.hist(combi['CODE_GENDER'], bins = 5, color = 'blue')
	plt.title('Male and Female loan applicants')
	plt.xlabel('Gender')
	plt.ylabel('population')
	plt.show()

.. image:: ../images/1.png
    :width: 49 %



To visualize the repartition of gender

.. code-block:: 

	plt.hist(combi['NAME_FAMILY_STATUS'], bins = 5, color = 'brown')
	plt.title('Marraige Statu loan applicants')
	plt.xlabel('Marraige Status')
	plt.ylabel('population')
	plt.show()

.. image:: ../images/2.png
    :width: 49 %

To show the repartition of the family status  

