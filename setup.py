import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="extensisq",
    version="0.6.0",
    author="W.R. Kampinga",
    author_email='wrkampi@tuta.io',
    description="Extend scipy.integrate with various methods for solve_ivp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WRKampi/extensisq",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=2.2.0",
        "scipy>=1.15.0",
    ],
    keywords=[
        'ode', 'ode-solver', 'ivp', 'ivp-methods', 'scipy', 'scipy-integrate',
        'runge-kutta', 'runge-kutta-nystrom', 'differential-equations',
        'cash-karp', 'prince', 'bogacki-shampine', 'adams', 'shampine-gordon',
        'adams-bashforth-moulton', 'ode113', 'predictor-corrector', 'solver',
        'sensitivity', 'sensitivity-analysis', 'trbdf2', 'trx2', 'esdirk',
        'dae'
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
    python_requires='>=3.10',
    tests_require=['pytest']
)
