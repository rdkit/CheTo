from setuptools import setup

setup(name='ChemTopicModel',
      version='0.2',
      description='Applying topic modeling for chemistry. https://dx.doi.org/10.1021/acs.jcim.7b00249',
      url='http://rdkit.org',
      author='Nadine Schneider',
      author_email='nadine.schneider.shb@gmail.com',
      license='BSD',
      packages=['ChemTopicModel'],
      install_requires=['scikit-learn'],
      zip_safe=False)
