import setuptools

setuptools.setup(
      name='trackdays',
      version='0.1.5',
      description='An RL environment and training code for a car on a racetrack.',
      url='http://github.com/hr0nix/trackdays',
      author='Boris Yangel',
      author_email='boris.jangel@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
            'tensorflow',
            'tf-agents',
            'highway-env @ git+https://github.com/eleurent/highway-env'
      ],
)
