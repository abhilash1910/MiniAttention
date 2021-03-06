# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:02:43 2020

@author: Abhilash
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:45:49 2020

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'MiniAttention',         
  packages = ['MiniAttention'],   
  version = '0.1',       
  license='MIT',        
  description = 'A mini-Hierarchical Attention Layer built for Document classification compatible with Keras and Tensorflow',   
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/MiniAttention',   
  download_url = 'https://github.com/abhilash1910/MiniAttention/archive/v_01.tar.gz',    
  keywords = ['Document Classification','Attention Layer','Heirarchical Attention','Word Level Attention','Keras','Tensorflow'],   
  install_requires=[           

          'numpy',         
          'keras',
          'tensorflow',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',

    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
