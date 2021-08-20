from setuptools import setup

setup(
    name='gym_fem',
    version='0.1',
    keywords='rl, environment, openaigym, openai-gym, gym, optimal control, finite element method',
    author="Johannes Dornheim",
    author_email='johannes.dornheim@hs-karlsruhe.de',
    install_requires=[
        'gym>=0.12',
        'pandas>=0.23',
        'numpy>=1.15',
        'scipy>=1.1',
        'imageio >= 2',
        'scikit-learn >= 0.20', 'matplotlib', 'pyglet'
    ],
    packages=["gym_fem", "gym_fem.envs", "gym_fem.examples.agents"],
    include_package_data=True
)
