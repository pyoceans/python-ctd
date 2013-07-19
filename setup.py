#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

try:  # Python 3
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2
    from distutils.command.build_py import build_py

from ctd import __version__ as version

install_requires = ['numpy', 'scipy', 'matplotlib', 'pandas', 'gsw']
url = 'http://pypi.python.org/packages/source'

classifiers = """\
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
License :: OSI Approved :: MIT License
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Education
Topic :: Software Development :: Libraries :: Python Modules
"""

README = open('README.md').read()
CHANGES = open('CHANGES.txt').read()

config = dict(name='ctd',
              version=version,
              packages=['ctd', 'ctd.test'],
              license=open('LICENSE').read(),
              long_description='%s\n\n%s' % (README, CHANGES),
              classifiers=filter(None, classifiers.split("\n")),
              description='Tools to load hydrographic data as DataFrames',
              author='Filipe Fernandes',
              author_email='ocefpaf@gmail.com',
              maintainer='Filipe Fernandes',
              maintainer_email='ocefpaf@gmail.com',
              url='http://pypi.python.org/pypi/ctd/',
              download_url='%s/c/ctd/ctd-%s.tar.gz' % (url, version),
              platforms='any',
              cmdclass={'build_py': build_py},
              keywords=['oceanography', 'data analysis', 'cnv', 'DataFrame'],
              zip_safe=False,
              include_package_data=True,
              install_requires=install_requires)

setup(**config)
