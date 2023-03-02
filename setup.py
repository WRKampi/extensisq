import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="extensisq",
    version="0.4.0",
    author="W.R. Kampinga",
    author_email='wrkampi@tuta.io',
    description="Extend scipy.integrate with various methods for solve_ivp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WRKampi/extensisq",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
    ],
    keywords=[
        'ode', 'ode-solver', 'ivp', 'ivp-methods', 'scipy', 'scipy-integrate',
        'runge-kutta', 'differential-equations', 'cash-karp', 'prince',
        'bogacki-shampine', 'adams', 'adams-bashforth-moulton', 'ode113',
        'predictor-corrector', 'shampine-gordon', 'solver', 'sensitivity',
        'sensitivity-analysis'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.8',
    tests_require=['pytest']
)
